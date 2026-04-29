"""
train_gpt_h.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import argparse
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

optimizer_code_path = Path(__file__).with_name("optimizers.py")
if optimizer_code_path.exists():
    code += "\n\n" + "#" * 100 + "\n# records/track_3_optimization/optimizers.py\n" + "#" * 100 + "\n"
    code += optimizer_code_path.read_text()


def parse_args():
    parser = argparse.ArgumentParser(description="Track 3 H-optimizer benchmark runner")
    parser.add_argument("--optimizer", choices=["muon", "muonh", "adamh", "adamw"], default="muon")
    parser.add_argument("--train-steps", type=int, default=3550)
    parser.add_argument("--eval-interval", type=int, default=125)
    parser.add_argument("--val-tokens", type=int, default=20 * 524288)
    parser.add_argument("--target-loss", type=float, default=3.28)
    parser.add_argument("--stop-at-target", action="store_true")
    parser.add_argument("--checkpoint-out", type=str, default=None,
                        help="If set, rank 0 saves the final model state_dict to this path at end of training.")
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--h-cooldown-frac", type=float, default=None)
    parser.add_argument("--aux-cooldown-frac", type=float, default=None)
    parser.add_argument("--h-min-lr-frac", type=float, default=0.0)
    parser.add_argument("--aux-min-lr-frac", type=float, default=0.0)
    parser.add_argument("--h-warmup-steps", type=int, default=0)
    parser.add_argument("--aux-warmup-steps", type=int, default=0)

    parser.add_argument("--matrix-lr", type=float, default=None)
    parser.add_argument("--matrix-weight-decay", type=float, default=0.01)
    parser.add_argument("--zero-muon-lr", type=float, default=0.02)
    parser.add_argument("--zero-muon-weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    # Per-module LR multipliers (applied to matrix_lr for h-path, and to zero_muon_lr for proj-path).
    parser.add_argument("--qkv-lr-mult", type=float, default=1.0)
    parser.add_argument("--mlp-fc-lr-mult", type=float, default=1.0)
    parser.add_argument("--attn-proj-lr-mult", type=float, default=1.0)
    parser.add_argument("--mlp-proj-lr-mult", type=float, default=1.0)

    parser.add_argument("--head-lr", type=float, default=1 / 320)
    parser.add_argument("--embed-lr", type=float, default=0.3)
    parser.add_argument("--scalar-lr", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.8)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--adamw-lr", type=float, default=0.0015)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.125)
    parser.add_argument("--adamw-warmup-steps", type=int, default=0)

    parser.add_argument("--h-beta1", type=float, default=0.9)
    parser.add_argument("--h-beta2", type=float, default=0.95)
    parser.add_argument("--h-eps", type=float, default=1e-8)
    parser.add_argument("--h-hyperball-eps", type=float, default=1e-10)

    parser.add_argument("--attn-proj-init", choices=["zero", "default"], default="zero")
    parser.add_argument("--mlp-proj-init", choices=["zero", "default"], default="zero")
    parser.add_argument("--head-init", choices=["zero", "default"], default="zero")
    parser.add_argument("--attn-proj-init-scale", type=float, default=1.0)
    parser.add_argument("--mlp-proj-init-scale", type=float, default=1.0)
    parser.add_argument("--head-init-scale", type=float, default=1.0)
    parser.add_argument("--qkv-init-scale", type=float, default=1.0)
    parser.add_argument("--mlp-fc-init-scale", type=float, default=1.0)
    # LayerScale-style proj: the proj output is multiplied by a learnable scalar
    # initialised to `--layerscale-init` (default 0). Combined with `--attn-proj-init default`
    # / `--mlp-proj-init default`, the matrix is regularly initialised but the residual
    # contribution starts at 0, letting MuonH operate on a nonzero matrix from step 1.
    parser.add_argument("--layerscale-proj", action="store_true")
    parser.add_argument("--layerscale-init", type=float, default=0.0)
    parser.add_argument("--debug-nan", action="store_true")
    parser.add_argument("--debug-every", type=int, default=1)
    parser.add_argument("--debug-max-reports", type=int, default=8)
    parser.add_argument("--debug-stop-step", type=int, default=None)
    return parser.parse_args()


args = parse_args()
if args.h_cooldown_frac is None:
    args.h_cooldown_frac = args.cooldown_frac
if args.aux_cooldown_frac is None:
    args.aux_cooldown_frac = args.cooldown_frac


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
    def __init__(self, dim: int, head_dim=128, layerscale: bool = False, layerscale_init: float = 0.0):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        self.layerscale = layerscale
        if layerscale:
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
        if self.layerscale:
            # Multiply in fp32. The bf16 backward of `y_bf16 * scale_bf16` accumulates the
            # per-element gradient w.r.t. `scale` over all (B, T) positions in a bf16
            # accumulator, which overflows past ±3.4e38 and produces NaN/inf in proj_scale.grad.
            y = (y.float() * self.proj_scale).type_as(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, layerscale: bool = False, layerscale_init: float = 0.0):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.layerscale = layerscale
        if layerscale:
            self.proj_scale = nn.Parameter(torch.full((dim,), float(layerscale_init)))

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.proj(x)
        if self.layerscale:
            # See `CausalSelfAttention.forward` — the bf16 multiply has a backward-accumulator
            # overflow on the per-channel sum; force fp32 for the LayerScale multiplication.
            x = (x.float() * self.proj_scale).type_as(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, layerscale: bool = False, layerscale_init: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(dim, layerscale=layerscale, layerscale_init=layerscale_init)
        self.mlp = MLP(dim, layerscale=layerscale, layerscale_init=layerscale_init)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, layerscale: bool = False, layerscale_init: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, layerscale=layerscale, layerscale_init=layerscale_init) for _ in range(num_layers)])
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

try:
    from records.track_3_optimization.optimizers import AdamH, Muon, MuonH, split_nonzero_and_zero_params
except ModuleNotFoundError:
    from optimizers import AdamH, Muon, MuonH, split_nonzero_and_zero_params


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


def report_nonfinite_tensor(kind: str, name: str, tensor: Tensor, step: int):
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return False
    finite_count = int(finite.sum().item())
    total = tensor.numel()
    stats = f"finite={finite_count}/{total}"
    if finite_count:
        values = tensor.detach()[finite].float()
        stats += (
            f" min={values.min().item():.6e}"
            f" max={values.max().item():.6e}"
            f" norm={values.norm().item():.6e}"
        )
    print(
        f"debug_nonfinite rank={dist.get_rank()} step={step} kind={kind}"
        f" name={name} shape={tuple(tensor.shape)} {stats}",
        flush=True,
    )
    return True


def report_named_nonfinite(kind: str, named_tensors, step: int):
    found = False
    printed = 0
    suppressed = 0
    for name, tensor in named_tensors:
        if printed < args.debug_max_reports:
            if report_nonfinite_tensor(kind, name, tensor, step):
                found = True
                printed += 1
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        found = True
        suppressed += 1
    if suppressed:
        print(
            f"debug_nonfinite_suppressed rank={dist.get_rank()} step={step}"
            f" kind={kind} count={suppressed}",
            flush=True,
        )
    return found


def sync_debug_flag(local_found: bool):
    flag = torch.tensor([int(local_found)], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())

# we begin by logging this file itself
print0(code, console=True)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("="*100)

val_tokens = args.val_tokens
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768,
            layerscale=args.layerscale_proj, layerscale_init=args.layerscale_init).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

train_steps = args.train_steps


def apply_init_mode_(param: Tensor, mode: str, scale: float):
    if mode == "zero":
        param.data.zero_()
    elif scale != 1.0:
        param.data.mul_(scale)


for name, p in model.named_parameters():
    if name.endswith("attn.proj.weight"):
        apply_init_mode_(p, args.attn_proj_init, args.attn_proj_init_scale)
    elif name.endswith("mlp.proj.weight"):
        apply_init_mode_(p, args.mlp_proj_init, args.mlp_proj_init_scale)
    elif name == "proj.weight":
        apply_init_mode_(p, args.head_init, args.head_init_scale)
    elif name.endswith(("attn.q.weight", "attn.k.weight", "attn.v.weight")):
        if args.qkv_init_scale != 1.0:
            p.data.mul_(args.qkv_init_scale)
    elif name.endswith("mlp.fc.weight"):
        if args.mlp_fc_init_scale != 1.0:
            p.data.mul_(args.mlp_fc_init_scale)
    elif "proj" in name:
        p.data.zero_()
    dist.broadcast(p.detach(), 0)

named_params = list(model.named_parameters())
param_names_by_id = {id(param): name for name, param in named_params}
hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_h_params, hidden_zero_params = split_nonzero_and_zero_params(hidden_matrix_params)
embed_params = [model.embed.weight]
head_params = [model.proj.weight]
scalar_params = [p for p in model.parameters() if p.ndim < 2]


def _classify_hidden(params):
    qkv, mlp_fc, attn_proj, mlp_proj, other = [], [], [], [], []
    for p in params:
        n = param_names_by_id[id(p)]
        if n.endswith(("attn.q.weight", "attn.k.weight", "attn.v.weight")):
            qkv.append(p)
        elif n.endswith("mlp.fc.weight"):
            mlp_fc.append(p)
        elif n.endswith("attn.proj.weight"):
            attn_proj.append(p)
        elif n.endswith("mlp.proj.weight"):
            mlp_proj.append(p)
        else:
            other.append(p)
    return qkv, mlp_fc, attn_proj, mlp_proj, other

matrix_lr = args.matrix_lr
if matrix_lr is None:
    matrix_lr = 0.02 if args.optimizer == "muon" else (0.02 * 0.01) ** 0.5

adam_aux_optimizer = AdamW(
    [
        dict(params=embed_params, lr=args.embed_lr),
        dict(params=head_params, lr=args.head_lr),
        dict(params=scalar_params, lr=args.scalar_lr),
    ],
    betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_eps,
    weight_decay=0,
    fused=True,
)
for group in adam_aux_optimizer.param_groups:
    group["schedule_type"] = "aux"

if args.optimizer == "muon":
    matrix_optimizer = Muon(
        hidden_matrix_params,
        lr=matrix_lr,
        weight_decay=args.matrix_weight_decay,
        momentum=args.muon_momentum,
    )
    for group in matrix_optimizer.param_groups:
        group["schedule_type"] = "aux"
    optimizers = [adam_aux_optimizer, matrix_optimizer]
elif args.optimizer == "muonh":
    optimizers = [adam_aux_optimizer]
    qkv_h, mlp_fc_h, attn_proj_h, mlp_proj_h, other_h = _classify_hidden(hidden_h_params)
    qkv_z, mlp_fc_z, attn_proj_z, mlp_proj_z, other_z = _classify_hidden(hidden_zero_params)
    h_groups = [
        ("qkv", qkv_h, args.qkv_lr_mult),
        ("mlp_fc", mlp_fc_h, args.mlp_fc_lr_mult),
        ("attn_proj", attn_proj_h, args.attn_proj_lr_mult),
        ("mlp_proj", mlp_proj_h, args.mlp_proj_lr_mult),
        ("other", other_h, 1.0),
    ]
    for label, ps, mult in h_groups:
        if not ps:
            continue
        opt = MuonH(
            ps,
            lr=matrix_lr * mult,
            momentum=args.muon_momentum,
            hyperball_eps=args.h_hyperball_eps,
        )
        for group in opt.param_groups:
            group["schedule_type"] = "h"
            group["module_label"] = label
        optimizers.append(opt)
    z_groups = [
        ("qkv_z", qkv_z, args.qkv_lr_mult),
        ("mlp_fc_z", mlp_fc_z, args.mlp_fc_lr_mult),
        ("attn_proj_z", attn_proj_z, args.attn_proj_lr_mult),
        ("mlp_proj_z", mlp_proj_z, args.mlp_proj_lr_mult),
        ("other_z", other_z, 1.0),
    ]
    for label, ps, mult in z_groups:
        if not ps:
            continue
        opt = Muon(
            ps,
            lr=args.zero_muon_lr * mult,
            weight_decay=args.zero_muon_weight_decay,
            momentum=0.95,
        )
        for group in opt.param_groups:
            group["schedule_type"] = "aux"
            group["module_label"] = label
        optimizers.append(opt)
elif args.optimizer == "adamh":
    optimizers = [adam_aux_optimizer]
    if hidden_h_params:
        h_optimizer = AdamH(
            hidden_h_params,
            lr=matrix_lr,
            betas=(args.h_beta1, args.h_beta2),
            eps=args.h_eps,
            hyperball_eps=args.h_hyperball_eps,
        )
        for group in h_optimizer.param_groups:
            group["schedule_type"] = "h"
        optimizers.append(h_optimizer)
    if hidden_zero_params:
        zero_optimizer = Muon(
            hidden_zero_params,
            lr=args.zero_muon_lr,
            weight_decay=args.zero_muon_weight_decay,
            momentum=0.95,
        )
        for group in zero_optimizer.param_groups:
            group["schedule_type"] = "aux"
        optimizers.append(zero_optimizer)
elif args.optimizer == "adamw":
    adamw_optimizer = AdamW(
        [dict(params=list(model.parameters()), lr=args.adamw_lr)],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.adamw_weight_decay,
        fused=True,
    )
    for group in adamw_optimizer.param_groups:
        group["schedule_type"] = "aux"
        group["warmup_steps"] = args.adamw_warmup_steps
    optimizers = [adamw_optimizer]
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

print0(f"optimizer:{args.optimizer} matrix_lr:{matrix_lr} h_cooldown_frac:{args.h_cooldown_frac}"
       + f" aux_cooldown_frac:{args.aux_cooldown_frac} h_min_lr_frac:{args.h_min_lr_frac}"
       + f" aux_min_lr_frac:{args.aux_min_lr_frac} nonzero_h_matrices:{len(hidden_h_params)}"
       + f" zero_default_matrices:{len(hidden_zero_params)} attn_proj_init:{args.attn_proj_init}"
       + f" mlp_proj_init:{args.mlp_proj_init} head_init:{args.head_init}"
       + f" attn_proj_init_scale:{args.attn_proj_init_scale} mlp_proj_init_scale:{args.mlp_proj_init_scale}"
       + f" head_init_scale:{args.head_init_scale}")
if args.debug_nan:
    h_names = [param_names_by_id[id(param)] for param in hidden_h_params]
    zero_names = [param_names_by_id[id(param)] for param in hidden_zero_params]
    print0("debug_h_param_names:" + ",".join(h_names), console=True)
    print0("debug_zero_default_param_names:" + ",".join(zero_names), console=True)
    if report_named_nonfinite("param_after_init", named_params, step=-1):
        print0("debug_nonfinite_after_init", console=True)


def get_lr(step: int, cooldown_frac: float, min_lr_frac: float, warmup_steps: int = 0):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        decay = 1.0
    else:
        decay = min_lr_frac + (1 - min_lr_frac) * (1 - progress) / cooldown_frac
    if warmup_steps > 0 and step < warmup_steps:
        decay *= (step + 1) / warmup_steps
    return decay


########################################
#        Training and Validation       #
########################################

training_time = 0
# start the clock
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):
    is_last_step = step == train_steps
    
    # --------------- VALIDATION SECTION -----------------
    if is_last_step or step % args.eval_interval == 0:
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
        local_debug_found = False
        if args.debug_nan:
            local_debug_found = report_nonfinite_tensor("val_loss", "validation", val_loss, step)
            if sync_debug_flag(local_debug_found):
                print0(f"debug_stop_nonfinite_validation step:{step}", console=True)
        reached_target = val_loss.item() <= args.target_loss
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()
        if reached_target:
            print0(f"target_loss_reached step:{step} target_loss:{args.target_loss:.5f} val_loss:{val_loss:.5f}",
                   console=True)
        if is_last_step or (args.stop_at_target and reached_target):
            break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    local_debug_found = False
    debug_this_step = args.debug_nan and step % args.debug_every == 0
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        if debug_this_step and report_nonfinite_tensor("train_loss", f"microbatch_{i}", loss, step):
            local_debug_found = True
        loss.backward()
    if debug_this_step and report_named_nonfinite(
        "local_grad_before_allreduce",
        [(name, p.grad) for name, p in named_params if p.grad is not None],
        step,
    ):
        local_debug_found = True
    for name, p in named_params:
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    if debug_this_step and report_named_nonfinite(
        "grad_after_allreduce",
        [(name, p.grad) for name, p in named_params],
        step,
    ):
        local_debug_found = True
    # set optimization hyperparameters and take a step
    for opt in optimizers:
        for group in opt.param_groups:
            is_h_group = group.get("schedule_type") == "h"
            cooldown_frac = args.h_cooldown_frac if is_h_group else args.aux_cooldown_frac
            min_lr_frac = args.h_min_lr_frac if is_h_group else args.aux_min_lr_frac
            default_warmup = args.h_warmup_steps if is_h_group else args.aux_warmup_steps
            warmup_steps = group.get("warmup_steps", default_warmup)
            group["lr"] = group["initial_lr"] * get_lr(step, cooldown_frac, min_lr_frac, warmup_steps)
            group["debug_nan"] = debug_this_step
            group["debug_step"] = step
            group["debug_param_names"] = param_names_by_id
        opt.step()
    if debug_this_step and report_named_nonfinite("param_after_step", named_params, step):
        local_debug_found = True
    if args.debug_nan and sync_debug_flag(local_debug_found):
        print0(f"debug_stop_nonfinite_training step:{step}", console=True)
        break
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)
    if args.debug_stop_step is not None and step >= args.debug_stop_step:
        print0(f"debug_stop_requested step:{step}", console=True)
        break

# Diagnostic: log per-block proj_scale stats at end of training (only meaningful if --layerscale-proj).
if dist.get_rank() == 0:
    for layer_idx, block in enumerate(model.blocks):
        for slot in ("attn", "mlp"):
            mod = getattr(block, slot)
            ps = getattr(mod, "proj_scale", None)
            if ps is None:
                continue
            ps_f = ps.detach().float()
            print(
                f"final_proj_scale layer={layer_idx} slot={slot}"
                f" mean={ps_f.mean().item():.6f}"
                f" abs_mean={ps_f.abs().mean().item():.6f}"
                f" std={ps_f.std().item():.6f}"
                f" min={ps_f.min().item():.6f}"
                f" max={ps_f.max().item():.6f}"
                f" l2={ps_f.norm().item():.6f}",
                flush=True,
            )
    for layer_idx, block in enumerate(model.blocks):
        for slot in ("attn", "mlp"):
            mod = getattr(block, slot)
            w = mod.proj.weight.detach().float()
            print(
                f"final_proj_weight layer={layer_idx} slot={slot}"
                f" frobenius={w.norm().item():.6f}"
                f" mean={w.mean().item():.6f}"
                f" std={w.std().item():.6f}"
                f" abs_mean={w.abs().mean().item():.6f}",
                flush=True,
            )
    if args.checkpoint_out:
        import os as _os
        _os.makedirs(_os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "step": step,
            },
            args.checkpoint_out,
        )
        print(f"saved_checkpoint path={args.checkpoint_out} bytes={_os.path.getsize(args.checkpoint_out)}", flush=True)

dist.destroy_process_group()
