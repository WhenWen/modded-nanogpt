import json
import os
import pathlib
import re
import subprocess
import sys
import time
from pathlib import Path

import modal


APP_NAME = "track3-h-optimizer-sweep"
REMOTE_REPO = Path("/root/modded-nanogpt")
LOCAL_REPO = Path(os.environ.get("MODDED_NANOGPT_LOCAL_REPO", Path.cwd())).resolve()
DATA_MOUNT = REMOTE_REPO / "data" / "fineweb10B"
RESULTS_PATH = Path("records/track_3_optimization/modal_results.jsonl")


def _ignore_upload(path: Path) -> bool:
    ignored = {".git", "logs", "__pycache__", ".pytest_cache", ".mypy_cache", "notes", "sweep_logs"}
    if any(part in ignored for part in path.parts):
        return True
    return "fineweb10B" in path.parts


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("track3-fineweb10b", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("track3-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = pathlib.Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(LOCAL_REPO, remote_path=str(REMOTE_REPO), ignore=_ignore_upload)
)


def _run_shell(cmd, cwd=REMOTE_REPO, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def _parse_training_output(output: str, target_loss: float):
    val_matches = []
    loss_pattern = r"[-+]?(?:nan|inf|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    for match in re.finditer(rf"step:(\d+)/(\d+) val_loss:({loss_pattern}).*?train_time:([0-9.]+)s", output):
        step = int(match.group(1))
        total_steps = int(match.group(2))
        val_loss = float(match.group(3))
        train_time = float(match.group(4))
        val_matches.append(
            {
                "step": step,
                "total_steps": total_steps,
                "val_loss": val_loss,
                "train_time": train_time,
            }
        )
    target_steps = [row["step"] for row in val_matches if row["val_loss"] <= target_loss]
    return {
        "target_step": min(target_steps) if target_steps else None,
        "final_val_loss": val_matches[-1]["val_loss"] if val_matches else None,
        "final_step": val_matches[-1]["step"] if val_matches else None,
        "val_history": val_matches,
    }


def _ensure_dataset(data_chunks: int):
    DATA_MOUNT.mkdir(parents=True, exist_ok=True)
    expected = DATA_MOUNT / f"fineweb_train_{data_chunks:06d}.bin"
    val = DATA_MOUNT / "fineweb_val_000000.bin"
    if expected.exists() and val.exists():
        return "dataset-cache-hit"
    code, output = _run_shell([sys.executable, "data/cached_fineweb10B.py", str(data_chunks)])
    if code != 0:
        raise RuntimeError(output)
    data_volume.commit()
    return "dataset-downloaded"


def _run_track3(config: dict):
    dataset_status = _ensure_dataset(int(config.get("data_chunks", 40)))
    gpu_count = int(
        subprocess.check_output("nvidia-smi -L | wc -l", shell=True, text=True).strip()
    )
    script_path = config.get("script_path", "records/track_3_optimization/train_gpt_h.py")
    args = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        script_path,
    ]
    if not config.get("self_contained", False):
        args += [
            "--optimizer", config["optimizer"],
            "--train-steps", str(config.get("train_steps", 3550)),
            "--eval-interval", str(config.get("eval_interval", 125)),
            "--val-tokens", str(config.get("val_tokens", 10485760)),
            "--target-loss", str(config.get("target_loss", 3.28)),
            "--cooldown-frac", str(config.get("cooldown_frac", 0.7)),
        ]
        if config.get("stop_at_target", True):
            args.append("--stop-at-target")
        if config.get("debug_nan", False):
            args.append("--debug-nan")
        if config.get("layerscale_proj", False):
            args.append("--layerscale-proj")
    optional_flags = {
        "h_cooldown_frac": "--h-cooldown-frac",
        "aux_cooldown_frac": "--aux-cooldown-frac",
        "h_min_lr_frac": "--h-min-lr-frac",
        "aux_min_lr_frac": "--aux-min-lr-frac",
        "h_warmup_steps": "--h-warmup-steps",
        "aux_warmup_steps": "--aux-warmup-steps",
        "matrix_lr": "--matrix-lr",
        "matrix_weight_decay": "--matrix-weight-decay",
        "muon_momentum": "--muon-momentum",
        "qkv_lr_mult": "--qkv-lr-mult",
        "mlp_fc_lr_mult": "--mlp-fc-lr-mult",
        "attn_proj_lr_mult": "--attn-proj-lr-mult",
        "mlp_proj_lr_mult": "--mlp-proj-lr-mult",
        "layerscale_init": "--layerscale-init",
        "head_lr": "--head-lr",
        "embed_lr": "--embed-lr",
        "scalar_lr": "--scalar-lr",
        "adam_beta1": "--adam-beta1",
        "adam_beta2": "--adam-beta2",
        "adam_eps": "--adam-eps",
        "adamw_lr": "--adamw-lr",
        "adamw_weight_decay": "--adamw-weight-decay",
        "adamw_warmup_steps": "--adamw-warmup-steps",
        "h_beta1": "--h-beta1",
        "h_beta2": "--h-beta2",
        "h_eps": "--h-eps",
        "h_hyperball_eps": "--h-hyperball-eps",
        "attn_proj_init": "--attn-proj-init",
        "mlp_proj_init": "--mlp-proj-init",
        "head_init": "--head-init",
        "attn_proj_init_scale": "--attn-proj-init-scale",
        "mlp_proj_init_scale": "--mlp-proj-init-scale",
        "head_init_scale": "--head-init-scale",
        "qkv_init_scale": "--qkv-init-scale",
        "mlp_fc_init_scale": "--mlp-fc-init-scale",
        "debug_every": "--debug-every",
        "debug_max_reports": "--debug-max-reports",
        "debug_stop_step": "--debug-stop-step",
    }
    if not config.get("self_contained", False):
        for key, flag in optional_flags.items():
            if key in config and config[key] is not None:
                args.extend([flag, str(config[key])])

    save_checkpoint = bool(config.get("save_checkpoint", False))
    checkpoint_filename = None
    if save_checkpoint and not config.get("self_contained", False):
        # filename is the run name + .pt; will live in /checkpoints volume
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", config["name"])[:200]
        checkpoint_filename = f"{safe_name}.pt"
        CHECKPOINT_MOUNT.mkdir(parents=True, exist_ok=True)
        args.extend(["--checkpoint-out", str(CHECKPOINT_MOUNT / checkpoint_filename)])

    started = time.time()
    code, output = _run_shell(args)
    if save_checkpoint:
        checkpoint_volume.commit()
    parsed = _parse_training_output(output, float(config.get("target_loss", 3.28)))
    capture_full = bool(config.get("capture_full_output", False))
    out_field = output if capture_full else output[-8000:]
    return {
        "name": config.get("name", config["optimizer"]),
        "config": config,
        "returncode": code,
        "elapsed_sec": round(time.time() - started, 3),
        "dataset_status": dataset_status,
        **parsed,
        "tail": out_field,
        "checkpoint_filename": checkpoint_filename if save_checkpoint else None,
    }


@app.function(image=image, gpu="H100:8", volumes={str(DATA_MOUNT): data_volume, str(CHECKPOINT_MOUNT): checkpoint_volume}, timeout=4 * 60 * 60)
def run_h100_8(config: dict):
    return _run_track3(config)


@app.function(image=image, gpu="H100", volumes={str(DATA_MOUNT): data_volume, str(CHECKPOINT_MOUNT): checkpoint_volume}, timeout=4 * 60 * 60)
def run_h100_1(config: dict):
    return _run_track3(config)


@app.function(image=image, gpu="A100:8", volumes={str(DATA_MOUNT): data_volume, str(CHECKPOINT_MOUNT): checkpoint_volume}, timeout=4 * 60 * 60)
def run_a100_8(config: dict):
    return _run_track3(config)


@app.function(image=image, gpu="A100", volumes={str(DATA_MOUNT): data_volume, str(CHECKPOINT_MOUNT): checkpoint_volume}, timeout=4 * 60 * 60)
def run_a100_1(config: dict):
    return _run_track3(config)


def _base_config(name, optimizer, train_steps, eval_interval, data_chunks, **kwargs):
    config = {
        "name": name,
        "optimizer": optimizer,
        "train_steps": train_steps,
        "eval_interval": eval_interval,
        "data_chunks": data_chunks,
        "target_loss": 3.28,
    }
    config.update(kwargs)
    return config


def _parse_float_list(values: str, default):
    if values:
        return [float(x) for x in values.split(",") if x.strip()]
    return default


def _parse_int_list(values: str):
    return [int(x) for x in values.split(",") if x.strip()]


def _parse_str_list(values: str, default):
    if values:
        return [x.strip() for x in values.split(",") if x.strip()]
    return default


def build_configs(
    preset: str,
    train_steps: int,
    eval_interval: int,
    data_chunks: int,
    matrix_lrs: str,
    cooldown_fracs: str,
    aux_cooldown_fracs: str,
    momentums: str,
    h_beta1s: str,
    h_beta2s: str,
    h_eps_values: str,
    h_min_lr_fracs: str,
    aux_min_lr_fracs: str,
    attn_proj_init: str = "",
    mlp_proj_init: str = "",
    head_init: str = "",
    attn_proj_init_scales: str = "",
    mlp_proj_init_scales: str = "",
    qkv_lr_mults: str = "",
    mlp_fc_lr_mults: str = "",
    attn_proj_lr_mults: str = "",
    mlp_proj_lr_mults: str = "",
):
    if matrix_lrs:
        lr_values = [float(x) for x in matrix_lrs.split(",") if x.strip()]
    else:
        lr_values = [0.008, 0.011, (0.02 * 0.01) ** 0.5, 0.018, 0.024, 0.032]
    cooldown_values = _parse_float_list(cooldown_fracs, [0.7, 0.8, 0.9, 0.95])
    aux_cooldown_values = _parse_float_list(aux_cooldown_fracs, [0.7])
    momentum_values = _parse_float_list(momentums, [0.95])
    h_beta1_values = _parse_float_list(h_beta1s, [0.9])
    h_beta2_values = _parse_float_list(h_beta2s, [0.95])
    h_eps_parsed = _parse_float_list(h_eps_values, [1e-8])
    h_min_lr_values = _parse_float_list(h_min_lr_fracs, [0.0])
    aux_min_lr_values = _parse_float_list(aux_min_lr_fracs, [0.0])

    if preset == "smoke":
        return [
            _base_config("smoke-muon", "muon", 1, 1, 1, val_tokens=65536),
            _base_config("smoke-muonh", "muonh", 1, 1, 1, matrix_lr=(0.02 * 0.01) ** 0.5, val_tokens=65536),
            _base_config("smoke-adamh", "adamh", 1, 1, 1, matrix_lr=(0.02 * 0.01) ** 0.5, val_tokens=65536),
        ]
    if preset == "baselines":
        return [
            _base_config("baseline-muon", "muon", train_steps, eval_interval, data_chunks),
            _base_config("baseline-adamw", "adamw", train_steps, eval_interval, data_chunks, adamw_warmup_steps=250),
        ]
    if preset == "baseline-muon":
        return [_base_config("baseline-muon", "muon", train_steps, eval_interval, data_chunks)]
    if preset == "baseline-adamw":
        return [_base_config("baseline-adamw", "adamw", train_steps, eval_interval, data_chunks, adamw_warmup_steps=250)]
    if preset == "h-lr-initial":
        configs = []
        for optimizer in ("muonh", "adamh"):
            for lr in lr_values:
                configs.append(_base_config(f"{optimizer}-lr-{lr:.6g}", optimizer, train_steps, eval_interval, data_chunks, matrix_lr=lr))
        return configs
    if preset in {"h-long-decay", "muonh-long-decay", "adamh-long-decay"}:
        optimizers = ("muonh", "adamh")
        if preset == "muonh-long-decay":
            optimizers = ("muonh",)
        elif preset == "adamh-long-decay":
            optimizers = ("adamh",)
        configs = []
        for optimizer in optimizers:
            for lr in lr_values:
                for cooldown_frac in cooldown_values:
                    for aux_cooldown_frac in aux_cooldown_values:
                        configs.append(
                            _base_config(
                                f"{optimizer}-lr-{lr:.6g}-hcd-{cooldown_frac:.2f}-auxcd-{aux_cooldown_frac:.2f}",
                                optimizer,
                                train_steps,
                                eval_interval,
                                data_chunks,
                                matrix_lr=lr,
                                h_cooldown_frac=cooldown_frac,
                                aux_cooldown_frac=aux_cooldown_frac,
                            )
                        )
        return configs
    if preset == "muonh-expanded":
        configs = []
        for lr in lr_values:
            for cooldown_frac in cooldown_values:
                for aux_cooldown_frac in aux_cooldown_values:
                    for momentum in momentum_values:
                        for h_min_lr_frac in h_min_lr_values:
                            for aux_min_lr_frac in aux_min_lr_values:
                                configs.append(
                                    _base_config(
                                        f"muonh-lr-{lr:.6g}-hcd-{cooldown_frac:.2f}-auxcd-{aux_cooldown_frac:.2f}-mom-{momentum:.3g}-hfloor-{h_min_lr_frac:.2f}-auxfloor-{aux_min_lr_frac:.2f}",
                                        "muonh",
                                        train_steps,
                                        eval_interval,
                                        data_chunks,
                                        matrix_lr=lr,
                                        h_cooldown_frac=cooldown_frac,
                                        aux_cooldown_frac=aux_cooldown_frac,
                                        muon_momentum=momentum,
                                        h_min_lr_frac=h_min_lr_frac,
                                        aux_min_lr_frac=aux_min_lr_frac,
                                    )
                                )
        return configs
    if preset == "adamh-expanded":
        configs = []
        for lr in lr_values:
            for cooldown_frac in cooldown_values:
                for aux_cooldown_frac in aux_cooldown_values:
                    for beta1 in h_beta1_values:
                        for beta2 in h_beta2_values:
                            for eps in h_eps_parsed:
                                for h_min_lr_frac in h_min_lr_values:
                                    for aux_min_lr_frac in aux_min_lr_values:
                                        configs.append(
                                            _base_config(
                                                f"adamh-lr-{lr:.6g}-hcd-{cooldown_frac:.2f}-auxcd-{aux_cooldown_frac:.2f}-b1-{beta1:.3g}-b2-{beta2:.3g}-eps-{eps:.0e}-hfloor-{h_min_lr_frac:.2f}-auxfloor-{aux_min_lr_frac:.2f}",
                                                "adamh",
                                                train_steps,
                                                eval_interval,
                                                data_chunks,
                                                matrix_lr=lr,
                                                h_cooldown_frac=cooldown_frac,
                                                aux_cooldown_frac=aux_cooldown_frac,
                                                h_beta1=beta1,
                                                h_beta2=beta2,
                                                h_eps=eps,
                                                h_min_lr_frac=h_min_lr_frac,
                                                aux_min_lr_frac=aux_min_lr_frac,
                                            )
                                        )
        return configs
    if preset == "muonh-init-grid":
        configs = []
        attn_modes = _parse_str_list(attn_proj_init, ["zero", "default"])
        mlp_modes = _parse_str_list(mlp_proj_init, ["zero", "default"])
        attn_scales = _parse_float_list(attn_proj_init_scales, [1.0])
        mlp_scales = _parse_float_list(mlp_proj_init_scales, [1.0])
        head_modes = _parse_str_list(head_init, ["zero"])
        for lr in lr_values:
            for cooldown_frac in cooldown_values:
                for aux_cooldown_frac in aux_cooldown_values:
                    for momentum in momentum_values:
                        for attn_mode in attn_modes:
                            for mlp_mode in mlp_modes:
                                for head_mode in head_modes:
                                    for attn_scale in attn_scales:
                                        for mlp_scale in mlp_scales:
                                            if attn_mode == "zero" and attn_scale != 1.0:
                                                continue
                                            if mlp_mode == "zero" and mlp_scale != 1.0:
                                                continue
                                            configs.append(
                                                _base_config(
                                                    f"muonh-init-lr-{lr:.6g}-hcd-{cooldown_frac:.2f}-auxcd-{aux_cooldown_frac:.2f}"
                                                    + f"-attn-{attn_mode}-{attn_scale:.3g}-mlp-{mlp_mode}-{mlp_scale:.3g}-head-{head_mode}",
                                                    "muonh",
                                                    train_steps,
                                                    eval_interval,
                                                    data_chunks,
                                                    matrix_lr=lr,
                                                    h_cooldown_frac=cooldown_frac,
                                                    aux_cooldown_frac=aux_cooldown_frac,
                                                    muon_momentum=momentum,
                                                    h_min_lr_frac=h_min_lr_values[0],
                                                    aux_min_lr_frac=aux_min_lr_values[0],
                                                    attn_proj_init=attn_mode,
                                                    mlp_proj_init=mlp_mode,
                                                    head_init=head_mode,
                                                    attn_proj_init_scale=attn_scale,
                                                    mlp_proj_init_scale=mlp_scale,
                                                )
                                            )
        return configs
    if preset == "muonh-record-3500":
        # Run the self-contained record file (records/track_3_optimization/train_gpt_simple_muonh.py).
        # Hardcoded to lr=0.018, h_cooldown_frac=0.99, train_steps=3500.
        cfg = _base_config(
            "muonh-record-3500",
            "muonh",  # nominal; ignored because self_contained=True
            3500,  # train_steps; ignored
            125,
            data_chunks,
        )
        cfg["self_contained"] = True
        cfg["script_path"] = "records/track_3_optimization/train_gpt_simple_muonh.py"
        cfg["capture_full_output"] = True
        return [cfg]
    if preset == "muonh-permod":
        qkv_mults = _parse_float_list(qkv_lr_mults, [1.0])
        mlpfc_mults = _parse_float_list(mlp_fc_lr_mults, [1.0])
        attnproj_mults = _parse_float_list(attn_proj_lr_mults, [1.0])
        mlpproj_mults = _parse_float_list(mlp_proj_lr_mults, [1.0])
        configs = []
        for lr in lr_values:
            for cooldown_frac in cooldown_values:
                for aux_cooldown_frac in aux_cooldown_values:
                    for qkv_m in qkv_mults:
                        for mlpfc_m in mlpfc_mults:
                            for attnproj_m in attnproj_mults:
                                for mlpproj_m in mlpproj_mults:
                                    configs.append(
                                        _base_config(
                                            f"muonh-permod-lr-{lr:.6g}-hcd-{cooldown_frac:.2f}-auxcd-{aux_cooldown_frac:.2f}"
                                            + f"-qkv-{qkv_m:.3g}-mlpfc-{mlpfc_m:.3g}-attnproj-{attnproj_m:.3g}-mlpproj-{mlpproj_m:.3g}",
                                            "muonh",
                                            train_steps,
                                            eval_interval,
                                            data_chunks,
                                            matrix_lr=lr,
                                            h_cooldown_frac=cooldown_frac,
                                            aux_cooldown_frac=aux_cooldown_frac,
                                            qkv_lr_mult=qkv_m,
                                            mlp_fc_lr_mult=mlpfc_m,
                                            attn_proj_lr_mult=attnproj_m,
                                            mlp_proj_lr_mult=mlpproj_m,
                                        )
                                    )
        return configs
    if preset == "single-muonh":
        lr = lr_values[0]
        h_cd = cooldown_values[0]
        aux_cd = aux_cooldown_values[0]
        return [_base_config(f"muonh-lr-{lr:.6g}-hcd-{h_cd:.2f}-auxcd-{aux_cd:.2f}",
                             "muonh", train_steps, eval_interval, data_chunks,
                             matrix_lr=lr, h_cooldown_frac=h_cd, aux_cooldown_frac=aux_cd)]
    if preset == "single-adamh":
        lr = lr_values[0]
        h_cd = cooldown_values[0]
        aux_cd = aux_cooldown_values[0]
        return [_base_config(f"adamh-lr-{lr:.6g}-hcd-{h_cd:.2f}-auxcd-{aux_cd:.2f}",
                             "adamh", train_steps, eval_interval, data_chunks,
                             matrix_lr=lr, h_cooldown_frac=h_cd, aux_cooldown_frac=aux_cd)]
    raise ValueError(f"Unknown preset: {preset}")


def _select_runner(gpu: str):
    if gpu == "h100-8":
        return run_h100_8
    if gpu == "h100-1":
        return run_h100_1
    if gpu == "a100-8":
        return run_a100_8
    if gpu == "a100-1":
        return run_a100_1
    raise ValueError(f"Unknown gpu preset: {gpu}")


@app.local_entrypoint()
def main(
    preset: str = "smoke",
    gpu: str = "h100-8",
    train_steps: int = 3550,
    eval_interval: int = 125,
    data_chunks: int = 40,
    matrix_lrs: str = "",
    cooldown_fracs: str = "",
    aux_cooldown_fracs: str = "",
    momentums: str = "",
    h_beta1s: str = "",
    h_beta2s: str = "",
    h_eps_values: str = "",
    h_min_lr_fracs: str = "",
    aux_min_lr_fracs: str = "",
    train_steps_values: str = "",
    eval_at_end_only: bool = False,
    stop_at_target: bool = True,
    attn_proj_init: str = "",
    mlp_proj_init: str = "",
    head_init: str = "",
    attn_proj_init_scales: str = "",
    mlp_proj_init_scales: str = "",
    attn_proj_init_scale: float = 1.0,
    mlp_proj_init_scale: float = 1.0,
    head_init_scale: float = 1.0,
    qkv_init_scale: float = 1.0,
    mlp_fc_init_scale: float = 1.0,
    debug_nan: bool = False,
    debug_every: int = 1,
    debug_max_reports: int = 8,
    debug_stop_step: int = -1,
    start_index: int = 0,
    max_runs: int = 0,
    parallel: int = 1,
    qkv_lr_mults: str = "",
    mlp_fc_lr_mults: str = "",
    attn_proj_lr_mults: str = "",
    mlp_proj_lr_mults: str = "",
    qkv_lr_mult: float = 1.0,
    mlp_fc_lr_mult: float = 1.0,
    attn_proj_lr_mult: float = 1.0,
    mlp_proj_lr_mult: float = 1.0,
    layerscale_proj: bool = False,
    layerscale_init: float = 0.0,
    capture_full_output: bool = False,
    save_checkpoint: bool = False,
    scalar_lr: float = -1.0,  # >= 0 → propagate as --scalar-lr; <0 → leave at script default
    embed_lr: float = -1.0,
    head_lr: float = -1.0,
):
    configs = build_configs(
        preset,
        train_steps,
        eval_interval,
        data_chunks,
        matrix_lrs,
        cooldown_fracs,
        aux_cooldown_fracs,
        momentums,
        h_beta1s,
        h_beta2s,
        h_eps_values,
        h_min_lr_fracs,
        aux_min_lr_fracs,
        attn_proj_init,
        mlp_proj_init,
        head_init,
        attn_proj_init_scales or (str(attn_proj_init_scale) if attn_proj_init_scale != 1.0 else ""),
        mlp_proj_init_scales or (str(mlp_proj_init_scale) if mlp_proj_init_scale != 1.0 else ""),
        qkv_lr_mults,
        mlp_fc_lr_mults,
        attn_proj_lr_mults,
        mlp_proj_lr_mults,
    )
    if train_steps_values:
        step_values = _parse_int_list(train_steps_values)
        expanded_configs = []
        for config in configs:
            for step_count in step_values:
                expanded = dict(config)
                expanded["train_steps"] = step_count
                expanded["name"] = f"{config['name']}-k-{step_count}"
                if eval_at_end_only:
                    expanded["eval_interval"] = step_count
                expanded["stop_at_target"] = stop_at_target
                expanded_configs.append(expanded)
        configs = expanded_configs
    else:
        for config in configs:
            if eval_at_end_only:
                config["eval_interval"] = config["train_steps"]
            config["stop_at_target"] = stop_at_target
    if preset != "muonh-init-grid":
        init_overrides = {
            "attn_proj_init": attn_proj_init or None,
            "mlp_proj_init": mlp_proj_init or None,
            "head_init": head_init or None,
            "attn_proj_init_scale": attn_proj_init_scale if attn_proj_init_scale != 1.0 else None,
            "mlp_proj_init_scale": mlp_proj_init_scale if mlp_proj_init_scale != 1.0 else None,
            "head_init_scale": head_init_scale if head_init_scale != 1.0 else None,
            "qkv_init_scale": qkv_init_scale if qkv_init_scale != 1.0 else None,
            "mlp_fc_init_scale": mlp_fc_init_scale if mlp_fc_init_scale != 1.0 else None,
        }
        name_suffix = "".join(f"-{key}-{value}" for key, value in init_overrides.items() if value is not None)
        for config in configs:
            for key, value in init_overrides.items():
                if value is not None:
                    config[key] = value
            if name_suffix:
                config["name"] += name_suffix
    if preset != "muonh-permod":
        permod_overrides = {
            "qkv_lr_mult": qkv_lr_mult if qkv_lr_mult != 1.0 else None,
            "mlp_fc_lr_mult": mlp_fc_lr_mult if mlp_fc_lr_mult != 1.0 else None,
            "attn_proj_lr_mult": attn_proj_lr_mult if attn_proj_lr_mult != 1.0 else None,
            "mlp_proj_lr_mult": mlp_proj_lr_mult if mlp_proj_lr_mult != 1.0 else None,
        }
        permod_suffix = "".join(f"-{key}-{value:.3g}" for key, value in permod_overrides.items() if value is not None)
        for config in configs:
            for key, value in permod_overrides.items():
                if value is not None:
                    config[key] = value
            if permod_suffix:
                config["name"] += permod_suffix
    if layerscale_proj:
        for config in configs:
            config["layerscale_proj"] = True
            config["layerscale_init"] = layerscale_init
            config["name"] += f"-layerscale-{layerscale_init:.3g}"
    if capture_full_output:
        for config in configs:
            config["capture_full_output"] = True
    if save_checkpoint:
        for config in configs:
            config["save_checkpoint"] = True
    if scalar_lr >= 0:
        for config in configs:
            config["scalar_lr"] = scalar_lr
            config["name"] += f"-scalar_lr-{scalar_lr:.3g}"
    if embed_lr >= 0:
        for config in configs:
            config["embed_lr"] = embed_lr
            config["name"] += f"-embed_lr-{embed_lr:.3g}"
    if head_lr >= 0:
        for config in configs:
            config["head_lr"] = head_lr
            config["name"] += f"-head_lr-{head_lr:.3g}"
    if debug_nan:
        for config in configs:
            config["debug_nan"] = True
            config["debug_every"] = debug_every
            config["debug_max_reports"] = debug_max_reports
            if debug_stop_step >= 0:
                config["debug_stop_step"] = debug_stop_step
    if start_index > 0:
        configs = configs[start_index:]
    if max_runs > 0:
        configs = configs[:max_runs]
    runner = _select_runner(gpu)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(configs), max(1, parallel)):
        batch = configs[start : start + max(1, parallel)]
        for config in batch:
            print(f"running {config['name']} on {gpu}: {json.dumps(config, sort_keys=True)}", flush=True)
        if parallel <= 1:
            results = [runner.remote(batch[0])]
        else:
            results = runner.map(batch, order_outputs=False, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"remote exception: {result!r}", flush=True)
                continue
            print(json.dumps({k: v for k, v in result.items() if k != "tail"}, sort_keys=True), flush=True)
            with RESULTS_PATH.open("a") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
            if result["returncode"] != 0:
                print(result["tail"], flush=True)
