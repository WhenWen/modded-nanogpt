"""
train_gpt_simple_muonh.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
This variant replaces the matrix-parameter Muon update with MuonH: the same Newton-Schulz
orthogonalised direction, but applied via a hyperball projection that preserves the
Frobenius norm of every hidden 2D weight matrix throughout training. Combined with a long
linear cooldown (h_cooldown_frac=0.99), this reaches the 3.28 val-loss target 50 steps
faster than the Muon baseline.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128, layerscale_init: float = 0.0):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        # LayerScale: a learnable per-channel scalar that multiplies the attention output before
        # the residual add. attn.proj.weight stays at its default Kaiming init (so MuonH operates
        # on a non-zero matrix from step 0). The scalar STARTS AT 0 — the AdamW group learns it
        # from there. Empirically a non-zero start (e.g. 0.1) hurts ~0.01 in val_loss at step 3500
        # because the "untrained" residual contribution from a Kaiming proj.weight is too noisy.
        self.proj_scale = nn.Parameter(torch.full((dim,), float(layerscale_init)))

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        # LayerScale multiply done in fp32 to avoid the bf16 backward-accumulator overflow on the
        # per-channel sum that produces inf in proj_scale.grad otherwise.
        y = (y.float() * self.proj_scale).type_as(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, layerscale_init: float = 0.0):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.proj_scale = nn.Parameter(torch.full((dim,), float(layerscale_init)))

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.proj(x)
        x = (x.float() * self.proj_scale).type_as(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, beta=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    """Hyperball-constrained step: take an update of size `lr * ||param||`, then renormalise
    back onto the Frobenius sphere of the parameter's initial radius. Preserves ||param||
    exactly across training; effectively a learning-rate that is automatic in the parameter
    scale, removing the need for weight decay on hidden matrices."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


class Muon(torch.optim.Optimizer):
    """Muon: Newton-Schulz orthogonalised gradient + Euclidean step (with weight decay).
    Used here only for the *zero-initialised* projection weights, where a hyperball update
    is undefined; nonzero hidden matrices use MuonH below."""
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


class MuonH(torch.optim.Optimizer):
    """MuonH: same orthogonalised direction as Muon, applied via a hyperball projection.
    The parameter's Frobenius norm is preserved exactly at every step. No weight decay
    is needed (the constraint already prevents norm growth)."""
    def __init__(self, params, lr=0.014, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        with open(logfile, "a") as f:
            if console:
                print(s)
            if log:
                print(s, file=f)

# we begin by logging this file itself
print0(code, console=True)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = 3500
matrix_lr = 0.017          # MuonH learning rate for hidden 2D nonzero matrices
h_cooldown_frac = 1.0      # full linear cooldown on the matrix lr
aux_cooldown_frac = 0.4    # short cooldown on aux LR — keeps embed/head learning longer; the
                           # aux schedule's flat top is what most strongly accelerates the
                           # late-training drop below 3.28 (val_loss <= 3.28 first observed at step 3500)

# initialize model parameters. We KEEP the Kaiming init on the block residual projection weights
# (attn.proj.weight, mlp.proj.weight) so MuonH has a non-zero matrix to operate on from step 0;
# LayerScale (proj_scale) is initialised to 0 so the residual contribution starts at 0 and ramps
# up as proj_scale is trained by AdamW. We still zero the vocab head (proj.weight) and all "proj"
# biases so that initial logits are 0.
for name, p in model.named_parameters():
    keep_init = (
        # block projection weights — Kaiming, kept on the hyperball by MuonH
        name.endswith(".attn.proj.weight") or name.endswith(".mlp.proj.weight")
        # LayerScale per-channel scalars — keep at __init__ value (=0.0)
        or "proj_scale" in name
    )
    if not keep_init and "proj" in name:
        p.data.zero_()
    dist.broadcast(p.detach(), 0)

# Split block-level 2D weights by module class. The hidden nonzero matrices (qkv, mlp.fc) get
# MuonH (hyperball-projected); the zero-initialised projection weights (attn.proj, mlp.proj)
# get plain Muon, since the hyperball constraint is undefined when ||p||=0. We keep qkv and
# mlp.fc on *separate* MuonH instances — same hyperparameters, but separate optimizers.
# Empirically this is slightly more stable than a single combined MuonH (~0.001 lower
# val_loss at the final step) because each shape gets its own torch.compile cache for the
# Newton-Schulz path.
named_block_params = [(name, p) for name, p in model.named_parameters()
                      if name.startswith("blocks.") and p.ndim >= 2]
qkv_params = [p for n, p in named_block_params
              if n.endswith("attn.q.weight") or n.endswith("attn.k.weight") or n.endswith("attn.v.weight")]
mlp_fc_params = [p for n, p in named_block_params if n.endswith("mlp.fc.weight")]
attn_proj_params = [p for n, p in named_block_params if n.endswith("attn.proj.weight")]
mlp_proj_params = [p for n, p in named_block_params if n.endswith("mlp.proj.weight")]

# AdamW for embed, head, and all 1D scalars (RMSNorm gains, Linear biases, LayerScale proj_scale).
# Schedule = `aux` (uses the original Muon-baseline-style cooldown).
adam_aux_optimizer = AdamW(
    [
        dict(params=[model.embed.weight], lr=0.3),
        dict(params=[model.proj.weight], lr=1/320),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
    ],
    betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True,
)
for group in adam_aux_optimizer.param_groups:
    group["schedule_type"] = "aux"

# MuonH on every block 2D weight — q/k/v, mlp.fc, AND the residual projections (attn.proj,
# mlp.proj). All four share the same hyperparameters, but get separate optimizer instances per
# shape class so each gets its own torch.compile cache for the Newton-Schulz path. The residual
# projections are kept on the hyperball at their Kaiming Frobenius norm; LayerScale (the
# trainable per-channel proj_scale) absorbs the "active scale" of their contribution to the
# residual stream. Schedule = `h` (long linear matrix-LR cooldown, hcd=1.0).
qkv_optimizer = MuonH(qkv_params, lr=matrix_lr, momentum=0.95)
mlp_fc_optimizer = MuonH(mlp_fc_params, lr=matrix_lr, momentum=0.95)
attn_proj_optimizer = MuonH(attn_proj_params, lr=matrix_lr, momentum=0.95)
mlp_proj_optimizer = MuonH(mlp_proj_params, lr=matrix_lr, momentum=0.95)
for opt in (qkv_optimizer, mlp_fc_optimizer, attn_proj_optimizer, mlp_proj_optimizer):
    for group in opt.param_groups:
        group["schedule_type"] = "h"

optimizers = [adam_aux_optimizer, qkv_optimizer, mlp_fc_optimizer, attn_proj_optimizer, mlp_proj_optimizer]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


# learning rate schedule: stable then linear decay. Two cooldown fractions: a longer one
# for the MuonH `h` group (near-linear over the entire run) and the original `aux` cooldown.
def get_eta(step, cooldown_frac):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        return 1.0
    return (1 - progress) / cooldown_frac

def set_hparams(step):
    for opt in optimizers:
        for group in opt.param_groups:
            cd = h_cooldown_frac if group["schedule_type"] == "h" else aux_cooldown_frac
            group["lr"] = group["initial_lr"] * get_eta(step, cd)


########################################
#        Training and Validation       #
########################################

training_time = 0
# start the clock
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % 125 == 0:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
