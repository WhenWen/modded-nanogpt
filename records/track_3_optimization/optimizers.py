import torch
from torch import Tensor
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _debug_tensor(label: str, tensor: Tensor, step, name: str, lr=None) -> bool:
    if bool(torch.isfinite(tensor).all().item()):
        return False
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total = tensor.numel()
    stats = f"finite={finite_count}/{total}"
    if finite_count:
        vals = tensor.detach()[finite].float()
        stats += (
            f" min={vals.min().item():.6e}"
            f" max={vals.max().item():.6e}"
            f" norm={vals.norm().item():.6e}"
        )
    lr_text = "" if lr is None else f" lr={lr:.6e}"
    print(f"debug_nonfinite rank={_rank()} step={step} name={name} tensor={label}{lr_text} {stats}", flush=True)
    return True


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update_impl(grad: Tensor, momentum: Tensor, beta=0.95, nesterov=True) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


_compiled_muon_update = torch.compile(_muon_update_impl)


def muon_update(grad: Tensor, momentum: Tensor, beta=0.95, nesterov=True) -> Tensor:
    if grad.is_cuda:
        return _compiled_muon_update(grad, momentum, beta=beta, nesterov=nesterov)
    return _muon_update_impl(grad, momentum, beta=beta, nesterov=nesterov)


@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10, assume_nonzero=False) -> None:
    """Hyperball-projected update.

    The step magnitude is `lr * p_norm * u_norm / (u_norm + eps)`, which is `lr * p_norm`
    when `u_norm >> eps` (the "fully scale-invariant" regime) and shrinks linearly with
    `u_norm` when `u_norm << eps` (so a tiny underlying update doesn't get inflated to
    a full-size hyperball step).

    With `eps = 1e-10` this matches the original `clamp(u_norm, min=1e-10)` behaviour for
    any non-trivial `u_norm`. To get the scale-aware behaviour, set `eps` to a value
    around the typical `u_norm` floor (e.g. `eps ≈ 1.0` for Adam-direction updates on a
    768×768 matrix, where typical `||u||_F ≈ sqrt(N)` per element).
    """
    p_norm = param.norm()
    u_norm = update.norm()
    if not assume_nonzero and float(p_norm) == 0.0:
        return
    new_param = param - lr * update * p_norm / (u_norm + eps)
    new_norm = new_param.norm() + eps
    param.copy_(new_param / new_norm * p_norm)


def split_nonzero_and_zero_params(params):
    nonzero_params = []
    zero_params = []
    for p in params:
        if p.ndim >= 2 and float(p.detach().norm()) == 0.0:
            zero_params.append(p)
        else:
            nonzero_params.append(p)
    return nonzero_params, zero_params


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        assert len(params) >= 1
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size() if _dist_ready() else 1
        rank = dist.get_rank() if _dist_ready() else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        continue
                    debug = group.get("debug_nan", False)
                    debug_names = group.get("debug_param_names", {})
                    debug_step = group.get("debug_step", None)
                    name = debug_names.get(id(p), "<unnamed>")
                    if debug:
                        _debug_tensor("grad_before_muon", p.grad, debug_step, name, group["lr"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    if debug:
                        _debug_tensor("momentum_after_muon", state["momentum_buffer"], debug_step, name, group["lr"])
                        _debug_tensor("update_after_muon", update, debug_step, name, group["lr"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    if debug:
                        _debug_tensor("param_after_muon", p, debug_step, name, group["lr"])
                if _dist_ready():
                    dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + rank])


class AdamH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.014,
        betas=(0.8, 0.95),
        eps=1e-8,
        hyperball_eps=1e-10,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, hyperball_eps=hyperball_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                update = (exp_avg / bias_correction1) / (exp_avg_sq / bias_correction2).sqrt().add(group["eps"])
                scale_invariant_update_(p, update, group["lr"], eps=group["hyperball_eps"], assume_nonzero=True)

        return loss


class MuonH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.014,
        momentum=0.95,
        nesterov=True,
        hyperball_eps=1e-10,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, hyperball_eps=hyperball_eps)
        params = list(params)
        assert len(params) >= 1
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size() if _dist_ready() else 1
        rank = dist.get_rank() if _dist_ready() else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        continue
                    debug = group.get("debug_nan", False)
                    debug_names = group.get("debug_param_names", {})
                    debug_step = group.get("debug_step", None)
                    name = debug_names.get(id(p), "<unnamed>")
                    if debug:
                        _debug_tensor("grad_before_muonh", p.grad, debug_step, name, group["lr"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    if debug:
                        _debug_tensor("momentum_after_muonh", state["momentum_buffer"], debug_step, name, group["lr"])
                        _debug_tensor("update_after_muonh", update, debug_step, name, group["lr"])
                    if debug and group.get("debug_print_update_norms", False):
                        print(
                            f"debug_h_update rank={_rank()} step={debug_step} name={name}"
                            f" lr={group['lr']:.6e} p_norm={p.norm().item():.6e}"
                            f" update_norm={update.norm().item():.6e}",
                            flush=True,
                        )
                    scale_invariant_update_(p, update, group["lr"], eps=group["hyperball_eps"], assume_nonzero=True)
                    if debug:
                        _debug_tensor("param_after_muonh", p, debug_step, name, group["lr"])
                if _dist_ready():
                    dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + rank])
