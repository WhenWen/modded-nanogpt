# MuonH tuning journal

Goal: minimize K such that val_loss ≤ 3.28 on the track-3 benchmark.

## Baseline
- Muon (record): K=3550 hits 3.28033 at step 3500 then 3.27940 at 3550. log = `c0ca36ae-1684-4362-aefb-c7654cf970ba.txt`.
- ~154 ms/step on H100×8 → ~10 min per full 3550-step run.

## Best so far (from `modal_results.jsonl` before this session)
- `lr=0.016, h_cd=0.97, aux_cd=0.70, mom=0.95` → reaches 3.28 at step **3500** (final 3.27967).
  At K=3550, final=3.27760 → 0.0024 below 3.28.
- baseline-muon at K=3750 reaches 3.28 at 3750 (re-run), official at 3550.
- AdamH variants: did not hit 3.28 in any tried configs.

## Code changes added this session
1. Added `a100-1` runner in `modal_sweep.py`.
2. Added `notes/`, `sweep_logs/` to ignore list (modal upload).
3. Per-module LR multipliers in `train_gpt_h.py`: `--qkv-lr-mult`, `--mlp-fc-lr-mult`, `--attn-proj-lr-mult`, `--mlp-proj-lr-mult`. Splits MuonH into per-module-class optimizers.
4. New preset `muonh-permod` for sweeping multipliers.
5. LayerScale architectural option in `train_gpt_h.py`: `--layerscale-proj` adds learnable `(dim,)` scalar that multiplies attn.proj and mlp.proj outputs (init = `--layerscale-init`, default 0). When combined with `--attn-proj-init default --mlp-proj-init default`, the proj weights are nonzero (Kaiming) so MuonH operates on them, while initial residual contribution is `0`.

Smoke confirmed both layerscale and the unchanged paths run on a100-1 (smoke_layerscale.log).

## Live sweeps (started 2026-04-27 ~ user sleep onset)
1. **Round 1** (`/tmp/track3_sweeps/round1.log`): preset=`muonh-expanded`, a100-1, parallel=8.
   `lr ∈ {0.014, 0.016, 0.018, 0.022}` × `h_cooldown ∈ {0.93, 0.97, 0.99}`, mom=0.95, aux_cd=0.70, K=3550, stop_at_target.
   12 configs total. ETA ≈ 89 min/run ≈ 3 hr wallclock.
2. **Round 2** (`/tmp/track3_sweeps/round2_permod.log`): preset=`muonh-permod`, a100-1, parallel=8.
   lr=0.016, hcd=0.97, qkv_mult ∈ {0.5, 1.0, 2.0}, mlp_fc_mult ∈ {0.5, 1.0, 2.0}, K=3550.
   9 configs total.
3. **Round 3** (`/tmp/track3_sweeps/round3_layerscale.log`): single-config probe.
   lr=0.016, hcd=0.97, layerscale_proj=true, attn/mlp proj_init=default. K=3550.
4. **Round 4** (`/tmp/track3_sweeps/round4_mom.log`): momentum sweep.
   lr=0.016, hcd=0.97, mom ∈ {0.85, 0.90, 0.93, 0.97, 0.98}. 5 configs.

Modal a100-1 is quota-limited to ~10 concurrent containers per workspace; later rounds queue behind earlier ones.

## ⚠️ Diagnosis update (~03:24 PDT)
- Round 1's first two completed configs both **NaN'd from step 125**:
  - `muonh-lr-0.014-hcd-0.93`: NaN at every checkpoint after step 0. elapsed=9720s.
  - `muonh-lr-0.016-hcd-0.93`: same. elapsed=9712s.
- a100-1 per-step time ~2.74s (matches 3550 × 2.74 ≈ 2.7 hr).
- Same configs (lr=0.016 hcd=0.93) ran **successfully on h100-8 before** (final 3.27838).
- Two open hypotheses:
  1. My per-module split breaks something — would also break on h100-8.
  2. bf16 NS or single-GPU all_gather is unstable on a100 — only breaks at world_size=1.
- **All a100-1 sweeps killed** (Rounds 1-4). Modal apps stopped via `app stop --yes`.
- h100-1 probe still running (will tell us if world_size=1 is the issue regardless of GPU vendor).
- **Launched h100-8 confirmation** of `lr=0.016 hcd=0.97` with my code (default mults). If success → my code is fine and the issue is a100/world_size=1; pivot all sweeps to h100-8.
- Wasted: ~2.7 hr × 8 a100-1 containers ≈ ~$25 on Round 1; ~$5-10 more across Rounds 2-4.

## Round 1 redux on h100-8 (in progress, started ~03:40 PDT)
preset=`muonh-expanded`, lr ∈ {0.016, 0.018, 0.020, 0.024} × hcd ∈ {0.97, 0.99}, parallel=4, K=3550, mom=0.95.

Confirmed h100-8 runs (~10–15 min/run):
| lr     | hcd  | target_step | final_val_loss | trajectory step 3500 |
| ------ | ---- | ----------- | -------------- | -------------------- |
| 0.016  | 0.97 | 3550        | 3.27892        | 3.28028              |
| 0.016  | 0.99 | **3500**    | **3.27828**    | 3.27828              |
| 0.016  | 0.97 (confirm) | 3500 | 3.27923   | 3.27923              |

**Current best:** `lr=0.016, hcd=0.99` — step 3500, final 3.27828.
- hcd=0.99 beats hcd=0.97 by ~0.001 final_val_loss; longer linear cooldown helps the late-training fine-tuning.
- baseline-muon at K=3550 was 3.27940 at step 3550 (final). MuonH at hcd=0.99 hits 3.28 at step 3500, not 3550.

## Round 2 redux on h100-8 (in progress, started ~03:41 PDT)
preset=`muonh-permod`, lr=0.016 hcd=0.97, qkv_lr_mult ∈ {0.7, 1.0, 1.5, 2.0} × mlp_fc_lr_mult ∈ {0.7, 1.0, 1.5}, parallel=4. 12 configs. No results yet.

## Round 3 LayerScale on h100-8 (started ~03:56 PDT)
Single probe: lr=0.016 hcd=0.97, layerscale_init=0.0, attn/mlp proj_init=default. **NaN from step 125 on h100-8 too.** (Same pattern as a100-1 NaN.) Likely the proj_scale=0 zeroes out proj.weight gradient → compile/numerical edge case. Skipping LayerScale; not worth more debug time tonight.

## Summary (so far) — best configs

| rank | lr    | hcd  | per-mod | target_step | final_val_loss |
| ---- | ----- | ---- | ------- | ----------- | -------------- |
| 1    | 0.018 | 0.99 | (1, 1)  | **3500**    | **3.27796**    |
| 2    | 0.016 | 0.99 | (1, 1)  | 3500        | 3.27828        |
| 3    | 0.018 | 0.97 | (1, 1)  | 3500        | 3.27833        |
| 4    | 0.020 | 0.99 | (1, 1)  | 3550        | 3.27854        |
| 5    | 0.017 | 0.99 | (1, 1)  | 3550        | 3.27876        |
| 6    | 0.016 | 0.97 | (1, 1)  | 3550        | 3.27892        |
| 7    | 0.020 | 0.97 | (1, 1)  | 3550        | 3.27960        |
| 8    | 0.019 | 0.99 | (1, 1)  | 3500        | 3.27987        |

Baseline-muon record: step 3550, final 3.27940. **Best MuonH improves by 50 steps** to 3.28 (3500 vs 3550).

## Round 5 per-mod 3×3 grid at lr=0.018 hcd=0.99 (h100-8, sequential due to capacity)
| qkv \ mlp_fc | 0.7 | 1.0 | 1.5 |
| ------------ | --- | --- | --- |
| 0.7          | 3.27923 (3550) | 3.27892 (3500) | 3.28913 (none) |
| 1.0          | 3.28147 (none) | **3.27796** (3500, R1 redux) | TBD |
| 1.5          | TBD | TBD | TBD |

Reductions in either qkv or mlp_fc consistently hurt. Increases pending.

## Action plan when R5 completes
- If a per-mod config beats 3.27796: confirm + finalize for record submission.
- Otherwise: best stays at lr=0.018 hcd=0.99 (qkv=1, mlp_fc=1). Submit as record.

## R5 RESULT — per-mod doesn't help

| qkv \ mlp_fc | 0.7 | 1.0 | 1.5 |
| ------------ | --- | --- | --- |
| 0.7 | 3.27923 (3550) | 3.27892 (3500) | 3.28913 (none) |
| 1.0 | 3.28147 (none) | **3.27796** (3500, R1) / 3.27953 (3550, R5 control) | 3.28691 (none) |
| 1.5 | 3.28612 (none) | 3.28187 (none) | 3.28902 (none) |

Every off-diagonal cell is worse than (1.0, 1.0). Mults > 1 effectively raise per-module LR past 0.020 (which we know fails). Mults < 1 effectively lower it past 0.014.

The (1.0, 1.0) diagonal also shows ~0.0016 run-to-run noise on `final_val_loss`.

## R6 K=3500 confirmation (lr=0.018, hcd=0.99)

```
config: optimizer=muonh matrix_lr=0.018 h_cooldown_frac=0.99 aux_cooldown_frac=0.70
        muon_momentum=0.95 K=3500 stop_at_target=true
val trajectory:
  step 3375: 3.28469
  step 3500: 3.27865    ← target hit
target_step = 3500   final_val_loss = 3.27865
```

**This is a record candidate**: 50 steps below baseline-muon (3500 vs 3550).

Existing supporting evidence:
- R1 redux at K=3550 with same hyperparams hit 3.28 at step 3500 (final 3.27796).
- R6 K=3500 explicitly stops at K=3500 with final 3.27865 (≤ 3.28).
- Two independent runs both reach 3.28 at step ≤ 3500.

## R7 logfile-capture run (running)
Same config, with `--capture-full-output` so the result tail field contains the full logfile (code + train log) for README submission.

## Submission plan
1. Wait for R7 logfile-capture to complete.
2. Save logfile to `records/track_3_optimization/<uuid>.txt`.
3. Update README.md to add a new row to results table:
   `| 2 | 3500 | MuonH (Muon + hyperball projection), lr=0.018 hcd=0.99 | 2026/04/27 | log | @kaiyuew |`

## DONE — Final state (~09:55 PDT 2026-04-27)
- Self-contained record file: `records/track_3_optimization/train_gpt_simple_muonh.py`.
- Logfile saved: `records/track_3_optimization/e842798a-a370-4d99-801c-2074b772e141.txt` (200 KB).
- README.md row 2 points to the new logfile.

The self-contained file is structured like `train_gpt_simple.py` — single file, hardcoded hparams, MuonH/Muon classes inline, no argparse. If someone replaces `train_gpt_simple.py` with this file and runs the quickstart command, the run reproduces (up to seed variance — see below).

### Seed-variance caveat
The self-contained file got two h100-8 runs with the exact same code+hparams: one missed (final 3.28167), one hit (final 3.27881). So `K=3500` is an "achievable" claim, not a "guaranteed" claim. If the user wants more rigour, run another 3-5 seeds and report median.

The earlier `train_gpt_h.py` derived runs showed the same: 3 of 3 K=3500 runs hit (3.27812, 3.27865, 3.27893). The split MuonH path in `train_gpt_h.py` may be slightly more stable (perhaps because of the per-module optimiser ordering interacting with NS bf16 numerics), but the difference is small enough to be noise. Both code paths reach K=3500 a majority of the time.
- Three independent K=3500 confirmations on h100-8:

| run | final_val_loss | target_step |
| --- | -------------- | ----------- |
| R6  | 3.27865        | 3500        |
| R6b | 3.27893        | 3500        |
| R7  | 3.27812        | 3500        |

All three reach 3.28 by step 3500 (median 3.27865, σ ≈ 0.0004). Improvement over baseline-muon: 50 steps.

### Caveats for the user (worth a follow-up before pushing)
1. The logfile is a concatenation of `train_gpt_h.py` + `optimizers.py` (with a separator line), and `train_gpt_h.py` still uses `from records.track_3_optimization.optimizers import ...`. Strictly per the README rule ("if we replace `train_gpt_simple.py` by the code, then running the quickstart will reproduce"), the file isn't self-runnable as `train_gpt_simple.py` — it'd need `optimizers.py` next to it. Two possible fixes:
   - Inline `optimizers.py` content into the script and delete the import.
   - Tell the README rule "logfile contains both files separated by a `### records/.../optimizers.py` marker".
2. The hyperparameters are still passed via CLI (the script uses argparse). The README rule prefers hardcoded. To make the run reproduce by plain quickstart, change defaults so the no-args run reproduces (set `--optimizer muonh`, `--matrix-lr 0.018`, `--h-cooldown-frac 0.99`, `--train-steps 3500` as defaults).
3. The contributor handle on the README row is `@kaiyue-wen` — replace with the correct GitHub handle if different.

### Things that did NOT help (so we don't retread them)
- Per-module LR multipliers (R5): 8 of 9 off-diagonal cells worse than (1, 1). Mults > 1 effectively raise per-module LR past 0.020 (which fails). Mults < 1 effectively lower it past 0.014 (also worse).
- LayerScale (R3 / R4): NaN by step 125 on both a100-1 and h100-8 with `attn_proj_init=default mlp_proj_init=default layerscale_proj=true layerscale_init=0.0`. Likely the proj_scale=0 zeroes proj.weight gradient and creates a numerical edge case. Not worth more debug.
- Higher LR (≥ 0.020): final loss above 3.28 at K=3550.
- Lower LR (≤ 0.014) at K=3550: hits later than step 3500.
- a100-1 muonh (any config) NaN'd from step 125 — bf16 NS or world_size=1 all_gather edge case. Use h100-8 only.

### LayerScale exploration (after fp32 fix) — all variants worse at step 3375

After fixing the bf16 backward overflow on `proj_scale.grad` (fp32 multiply for LayerScale), tried 6 LayerScale variants at K=3450, lr=0.018, hcd=0.99, aux_cd=0.4, mom=0.95:

| ls_init | proj_init_scale | scalar_lr | step 3375 | step 3450 (final) |
| ------- | --------------- | --------- | --------- | ----------------- |
| 0.0     | 1.0             | 0.01      | 3.28953   | 3.28582           |
| 0.01    | 0.5             | 0.01      | 3.29117   | 3.28745           |
| 0.1     | 1.0             | 0.01      | 3.28635   | 3.28263 *(best LS)* |
| 0.5     | 1.0             | 0.01      | 3.28820   | 3.28450           |
| 0.0     | 1.0             | 0.10      | 3.29316   | 3.28926           |

Compare to **no-LayerScale at same base** (lr=0.018, hcd=0.99, aux_cd=0.4, K=3450): step 3375 = **3.27975**, hits target. So *every* LayerScale variant tested misses step 3375 by 0.006-0.014.

**Interpretation.** With LayerScale, the residual contribution = `proj_scale * (Kaiming proj.weight @ x)`. proj_scale starts at `ls_init`, proj.weight is held at the Kaiming Frobenius norm (~38) by MuonH. For ls_init=0, the residual is exactly 0 at step 0 — the network is "frozen" early because every block's residual contribution is multiplied by 0. Even at ls_init=0.1 the contribution starts ~10× too small. By the time proj_scale grows to a useful magnitude (via a 1D-AdamW with `scalar_lr=0.01`, requires ~100s of steps), the no-LayerScale baseline (which starts proj.weight at 0 and grows it directly via plain Muon) has already done useful work.

scalar_lr=0.1 makes proj_scale ramp up faster but worse — proj_scale overshoots the relevant magnitude before proj.weight has rotated into a useful direction.

**Conclusion.** For this scale/setup, the existing zero-init-proj-plus-plain-Muon setup is empirically the right operating point; LayerScale-style "Kaiming proj + scalar" loses ~0.005-0.01 of val_loss at step 3375. Dropping LayerScale.

### Pushing below K=3500 — three attempts, all miss
At K=3475 (single seed each):
- lr=0.018 hcd=0.99 → 3.28347 (miss)
- lr=0.020 hcd=0.99 → 3.28163 (miss, but closest)
- lr=0.022 hcd=0.99 → 3.28477 (miss)

Higher LR helps slightly for shorter K but not enough. K=3475 needs ~0.0016 more progress per the typical curve, which the schedule can't deliver in 25 fewer steps without overshooting. Could try lr=0.020 with a different cooldown fraction, but the gap looks structural — K=3500 is likely the limit for this code path. Stopping here.

### Things still on the table (not tried thoroughly)
- Multi-seed averaging — single-run noise is ~0.0004 final_val_loss at this LR/hcd; need ≥3 seeds to call any small change real.
- LayerScale with `layerscale_init > 0` (e.g. 0.01) so the proj.weight gradient isn't zeroed at step 1.
- Aux LR / aux betas / aux cooldown sweep (untouched all session).
- Cosine cooldown shape vs the linear one used here.
- Higher momentum schedules (mom_increase_during_training).
- Warmup for matrix LR (added flag `--h-warmup-steps`, never swept).

## Planned next
- Round 4 (after Round 1 finishes): fine LR sweep around best from Round 1.
- Round 5 (after Round 2): refine per-module mults around best.
- Round 6: combine LayerScale + per-module mults.
- Round 7: confirm best config on h100-8 for record.

## Cost tracking
- Modal a100 at ~$1.10/hr. ~$30-50 expected for Rounds 1-3.

## R32-R34 (LayerScale ON + Kaiming-proj + hcd=1.0, hard constraints)

User's hard constraint: LayerScale ON, projection matrices NOT zero-init, h_cooldown_frac=1.0.

LR sweep at K=3500 hcd=1.0 aux_cd=0.4 mom=0.95 ls_on=true attn/mlp_proj=default Kaiming, ls_init=0.1 (in train_gpt_h.py — but see note below):

| lr    | seed | final_val_loss | step  |
| ----- | ---- | -------------- | ----- |
| 0.017 |   1  | **3.27849**    | 3500  | ← hit
| 0.017 |   2  | 3.28434        | 3500  | miss
| 0.018 |   1  | 3.28192        | 3500  | miss
| 0.018 |   2  | 3.2892         | 3500  | miss
| 0.020 |   -  | 3.28432 @ K=3550 | -    | miss

Best under hard constraints: lr=0.017 K=3500 → 3.27849, but 1/2 seeds. Single-seed noise dominates. lr=0.018 strictly worse.

**Init-bug in train_gpt_h.py**: the catch-all `elif "proj" in name: p.data.zero_()` in init also matches `proj_scale` (substring), so all `train_gpt_h.py` LayerScale runs effectively had `proj_scale=0` regardless of `--layerscale-init` value. The "working" config above had `proj_scale=0`, not 0.1.

R34 self-contained with explicit `proj_scale=0.1` (correct LayerScale): final=3.28957 (miss). Confirms that with `proj_scale=0.1` initial, the K=3500 hit is much harder. Aligning self-contained file to also init `proj_scale=0` (matching the working train_gpt_h.py behaviour).

## R35 — self-contained verification ✅ HIT (K=3500, final=3.27636)

Modified `train_gpt_simple_muonh.py`:
- `CausalSelfAttention.__init__` default `layerscale_init=0.0` (was 0.1).
- `MLP.__init__` default `layerscale_init=0.0` (was 0.1).
- Init comment updated: "proj_scale is initialised to 0".

Running `muonh-record-3500` preset on h100-8 → **final_val_loss=3.27636 at step 3500** (target 3.28 hit, buffer 0.004). Better than R33 (3.27849) — same nominal config but cleaner self-contained code path.

Logfile saved: `records/track_3_optimization/c1f30ba6-f8c8-453e-91ff-4177ac99c7ff.txt` (203 KB).

## R36 — push K below 3500 (in flight 21:42 PDT)

K-sweep at lr=0.017, hcd=1.0, aux_cd=0.4, mom=0.95, ls_on=true ls_init=0, attn/mlp_proj=default Kaiming, K ∈ {3475, 3450, 3425}. Single seed each as coarse probe. h100-8, sequential, expected wall-time per K ~25 min → all three lands by ~22:55 PDT. App: `ap-Z495BFTl9lgQlR0SQzdB3B`.

Hypothesis: R35's 0.004 buffer below target suggests K=3475 is plausible; K=3450 marginal; K=3425 likely miss (note: prior K=3450 record at hcd=0.99 no-LS hit 3.27592, so the same step count is reachable with the unconstrained config — the question is whether the constrained config can match it).

### R36 — all 3 configs crashed on flaky Modal node
All 3 configs of R36 died with transient CUDA errors (NCCL peer NVLink + invalid address space). Same node, rank 2 consistently the culprit. Re-launched as R37 on fresh allocation.

## R37 — K-sweep at lr=0.017 (clean) — all miss

| K    | final_val_loss | step  | margin |
| ---- | -------------- | ----- | ------ |
| 3500 | **3.27636**    | 3500  | hit    | (R35)
| 3475 | 3.28309        | 3475  | -0.003 |
| 3450 | 3.28223        | 3450  | -0.002 |
| 3425 | 3.28356        | 3425  | -0.004 |

All sub-3500 misses are within ~0.001 of each other and the trend isn't strictly monotone in K — single-seed noise (~0.0006 σ) dominates. With lr=0.017 the constrained config crosses 3.28 right around K=3500. Buffer at K=3500 (0.004) is enough that some sub-3500 K should hit on a lucky seed; needs higher LR or a different schedule shape to be reliable.

## R38 — higher-LR sweep at K∈{3475,3450} (in flight 22:46 PDT)

K ∈ {3475, 3450}, lr ∈ {0.018, 0.019, 0.020}, hcd=1.0, aux_cd=0.4, mom=0.95, ls_on=true ls_init=0, attn/mlp_proj=default Kaiming. 6 configs sequential on h100-8 → ~150 min, lands ~01:15 PDT 2026-04-28.

Hypothesis: bumping LR slightly should let the constrained config hit 3.28 in fewer steps. Prior baselines showed lr ≥ 0.020 overshoots at K=3550, but for K≤3475 the schedule is shorter so peak LR can be higher.

### R38 partial results then RemoteError (session gap 23:36→09:58 PDT 2026-04-28)

Completed before crash:
| K    | lr    | final_val_loss | step | margin |
| ---- | ----- | -------------- | ---- | ------ |
| 3475 | 0.018 | **3.28115**    | 3475 | -0.001 |
| 3450 | 0.018 | 3.28388        | 3450 | -0.004 |

LR bump from 0.017→0.018 helps K=3475 close the gap (3.28309→3.28115). K=3475 lr=0.018 is the closest sub-3500 result to date. R38 then crashed mid-flight (RemoteError); pool drained during 10h session gap.

## R39/R40/R41 — joint init_scale × lr sweep at K=3475 (in flight 09:59 PDT 2026-04-28)

Per user feedback "sweep different initialization std and lr jointly", launching 3 parallel modal apps:
- R39: scale=0.5, lr ∈ {0.018, 0.019, 0.020}, K=3475 (3 configs)
- R40: scale=1.5, lr ∈ {0.018, 0.019, 0.020}, K=3475 (3 configs)
- R41: scale=1.0, lr ∈ {0.019, 0.020}, K=3475 (2 configs — completes R38)

8 configs total. h100-8 pool ≈ 1 → sequential, ~25 min each, lands ~13:20 PDT. With existing scale=1.0 lr=0.018 result (3.28115), this gives a 3×3 lr×scale grid + extras at K=3475.

Hypothesis: scale<1 (smaller init) may hurt because residual contribution starts smaller and is amplified less by hyperball; scale>1 may overshoot or destabilize NS. Sweet spot likely scale∈[0.5, 1.5] with appropriate lr.

### R39/R40/R41 results — joint init_scale × lr grid at K=3475

| scale \ lr | 0.018      | 0.019    | 0.020    |
| ---------- | ---------- | -------- | -------- |
| 0.5        | 3.28651    | 3.28720  | 3.29175  |
| 1.0        | **3.28115**| 3.28685  | 3.28310  |
| 1.5        | 3.28125    | 3.28197  | 3.28400  |

- scale=0.5 row uniformly worst (smaller init hurts — residual contribution is too small early).
- scale=1.0 and scale=1.5 essentially tied at lr=0.018 (3.28115 vs 3.28125 — within seed noise).
- lr=0.018 is the LR sweet spot at both 1.0 and 1.5; lr=0.019 lr=0.020 worse.
- Best K=3475 result: 3.28115 (scale=1.0, lr=0.018) — miss by 0.001.

Joint init_scale × lr axis is essentially flat between 1.0 and 1.5 — init scale matters less than expected once it's not too small.

## R42/R43/R44 — extending to scale=2.0, finer LR, mom=0.97

Single-seed misses by 0.001 at K=3475 — likely some seeds hit. Trying new dimensions:
- R42: scale=1.0, lr ∈ {0.0175, 0.0185}, mom=0.95, K=3475 (finer LR around best)
- R43: scale=2.0, lr ∈ {0.018, 0.019}, mom=0.95, K=3475 (extend scale axis upward)
- R44: scale=1.0, lr ∈ {0.018, 0.019}, mom=0.97, K=3475 (higher momentum)

Results:
| run | scale | lr     | mom  | K    | final_val_loss | margin |
| --- | ----- | ------ | ---- | ---- | -------------- | ------ |
| R44 | 1.0   | 0.018  | 0.97 | 3475 | 3.28982        | -0.010 (mom=0.97 hurts a lot) |
| R44 | 1.0   | 0.019  | 0.97 | 3475 | 3.29012        | -0.010 |
| R42 | 1.0   | 0.0175 | 0.95 | 3475 | 3.28516        | -0.005 (worse than 0.017) |
| R42 | 1.0   | 0.0185 | 0.95 | 3475 | 3.28440        | -0.004 |
| **R43** | **2.0** | **0.018** | **0.95** | **3475** | **3.27947** | **+0.001 — HIT!** |

🎯 **First sub-3500 hit under hard constraints**: K=3475 with scale=2.0, lr=0.018, mom=0.95, hcd=1.0, ls_init=0. Scale axis IS productive — bigger init helps when residual contribution starts large enough to drive useful learning.

## R45/R46/R47 — push K + scale=2.5 + multi-seed

| run | scale | lr     | K    | final_val_loss | margin |
| --- | ----- | ------ | ---- | -------------- | ------ |
| R45 | 2.0   | 0.018  | 3450 | 3.28143        | -0.001 (miss, but tight) |
| R45 | 2.0   | 0.018  | 3425 | **3.27907**    | +0.001 (HIT) |
| R45 | 2.0   | 0.018  | 3400 | 3.28433        | -0.004 (miss, beyond noise) |
| R46 | 2.5   | 0.018  | 3475 | **3.27709**    | **+0.003 (HIT, biggest margin)** |
| R47 | 2.0   | 0.018  | 3475 | 3.28052        | -0.0005 (miss; seed 2 of K=3475 scale=2.0 lr=0.018 — first seed (R43) hit at 3.27947) |

scale=2.5 substantially better than scale=2.0 at K=3475 (margin 0.003 vs 0.001). The scale axis trend is still rising.

## R48/R49 — push K at scale=2.5 + scale=3.0

- R48: scale=2.5, lr=0.018, K ∈ {3450, 3425, 3400, 3375}
- R49: scale=3.0, lr ∈ {0.018, 0.019}, K=3475

R49 scale=3.0 K=3475: lr=0.018 → 3.27734 (HIT 0.0027), lr=0.019 → 3.27774 (HIT 0.0023). 2/2 at scale=3.0.

K=3475 hit map (val_loss ≤ 3.28 = HIT):

| scale \ lr | 0.018 | 0.019 |
| ---------- | ----- | ----- |
| 0.5 | 3.28651 miss | 3.28720 miss |
| 1.0 | 3.28115 miss | 3.28685 miss |
| 1.5 | 3.28125 miss | 3.28197 miss |
| 2.0 | 3.27947 HIT / 3.28052 miss (1/2) | 3.27968 HIT |
| 2.5 | **3.27709 HIT** | 3.28117 miss |
| 3.0 | 3.27734 HIT | 3.27774 HIT |

Best margin: scale=2.5 lr=0.018 (3.27709, +0.0029). Scale=3.0 most reliable (2/2).

K=3425 single seed at scale=2.0 lr=0.018: 3.27907 (HIT, +0.0009). K=3450 scale=2.0 lr=0.018 missed (3.28143, -0.0014).

## R50 — instrumented run for proj_scale convergence (in flight 14:53 PDT)

Goal (per user feedback): observe what `proj_scale` converges to, then absorb it into the projection init scale and **remove LayerScale entirely** while keeping MuonH on hidden matrices.

Added per-block proj_scale logging to `train_gpt_h.py`: at end of training prints mean/std/min/max/l2 of attn.proj_scale and mlp.proj_scale per block. Running K=3475 scale=2.5 lr=0.018 with `--capture-full-output` so we capture the values.

Plan after R50 lands:
1. Compute mean per-channel `proj_scale_conv` per slot (attn vs mlp).
2. Drop `--layerscale-proj`. Set `attn_proj_init_scale = 2.5 × mean(attn proj_scale_conv)` and similarly for mlp.
3. Run K=3475 with no LayerScale, same MuonH for projections, see if it matches the LayerScale-on results.

### NEW HARD CONSTRAINT (locked 15:14 PDT 2026-04-28)
"Remove layerscale and only using muonh should be a hard constraint." All record runs going forward must NOT use LayerScale. Instrumented LayerScale-on runs (R50b) only as a diagnostic to set the right `attn_proj_init_scale`.

R48 cancelled to free h100-8 queue for R50b. R50 cancelled (queued, never trained); replaced by R50b which adds checkpoint saving (Modal Volume `track3-checkpoints`) + per-block proj.weight Frobenius logging on top of the previous proj_scale logging.

## R50b — instrumented + checkpoint (in flight 15:16 PDT)

K=3475 scale=2.5 lr=0.018 (best LayerScale-on margin). Modifications applied to repo:
- `train_gpt_h.py`: log per-block (proj_scale, proj.weight) stats at end of training; save state_dict to `--checkpoint-out` if provided.
- `modal_sweep.py`: new `--save-checkpoint` flag; `track3-checkpoints` Modal Volume mounted at `/checkpoints` in run_* functions; passes `--checkpoint-out=/checkpoints/<sanitized-name>.pt`.

App: `ap-ARpTuc7wC7lYE4g49YYb1F`. Once it lands:
1. Read final `proj_scale` mean per (block × slot) from captured stdout.
2. Compute uniform-equivalent `attn_proj_init_scale ≈ 2.5 × mean(attn_proj_scale across blocks)` (and likewise mlp).
3. Launch follow-up: same K, same lr, drop `--layerscale-proj`, use computed scales. Should match LayerScale-on if absorption thesis is correct.

**R50b outcome:** training succeeded (val_loss 3.27724 — 3rd confirmation of K=3475 scale=2.5 lr=0.018 hit), but rc=1 because diagnostic block crashed with `NameError: name 'rank' is not defined` (used `rank` instead of `dist.get_rank()`). Proj_scale logging never ran. Fixed and re-launched as R50c.

**R50c outcome:** rc=0; final_val_loss=3.28399 (3rd seed of this config — MISS by 0.004; the other 2 seeds hit at 3.27709/3.27724, so 2/3 hit rate). Diagnostic + checkpoint successful. Checkpoint downloaded to `/tmp/track3_checkpoints/...K=3475...scale=2.5...layerscale-0.pt` (546MB).

### proj_scale convergence (R50c, K=3475 scale=2.5 lr=0.018)

`proj.weight` Frobenius is **fixed at 40** for all 24 (block × {attn, mlp}) — confirms hyperball preserves init norm exactly throughout training. Kaiming Frobenius ≈ 16 for 768→768 (and 3072→768 mlp.proj is also ≈16, by coincidence of dimensions).

`proj_scale` has near-zero mean (sign flips balanced) but large `abs_mean` per layer:

| layer | attn abs_mean | mlp abs_mean | attn F_eff | mlp F_eff | attn absorb-scale | mlp absorb-scale |
| ----- | ------------- | ------------ | ---------- | --------- | ----------------- | ---------------- |
| 0 | 0.13 | 0.18 | 5.5 | 8.0 | 0.34 | 0.50 |
| 5 | 0.24 | 0.37 | 10.4 | 15.6 | 0.65 | 0.98 |
| 10 | **1.39** | 0.52 | **57.0** | 21.8 | **3.56** | 1.36 |
| 11 | 0.88 | 0.37 | 36.2 | 15.7 | 2.27 | 0.98 |

`F_eff = ||proj_scale||_2 × ||proj.weight||_F / sqrt(768)` (assumes equal-norm rows of `proj.weight`; absorb-scale = F_eff / Kaiming_F=16).

**Across-layers averages:**
- attn: F_eff_avg = 20.06, absorbed_scale_avg = **1.25**
- mlp:  F_eff_avg = 15.75, absorbed_scale_avg = **0.98**

LayerScale is effectively doing per-layer scaling (especially attn — layer 10 needs 3.56× while layer 0 needs 0.34×). A single uniform absorbed scale won't capture this; experiment will show how much that matters.

## R51 — no-LayerScale absorbed experiment (in flight 16:05 PDT)

K ∈ {3475, 3500}, lr=0.018, mom=0.95, hcd=1.0, aux_cd=0.4. **No `--layerscale-proj`.** `attn_proj_init_scale=1.25, mlp_proj_init_scale=1.0`. App: `ap-k1XViTBOfTCLvt2Qn41NLn`. Single seed each ~50 min.

Hypothesis: if LayerScale's role is ~global magnitude scaling, R51 should hit at K=3475 (or close). If per-layer scale variation matters, R51 will miss; we'd then need per-layer init or accept ~K=3500 as the floor under the no-LayerScale hard constraint.

### R51 results — no-LayerScale absorbed init works at K=3475

| K    | attn_scale | mlp_scale | final_val_loss | margin |
| ---- | ---------- | --------- | -------------- | ------ |
| 3475 | 1.25       | 1.0       | **3.27958**    | +0.0004 (HIT) |
| 3500 | 1.25       | 1.0       | 3.27933        | +0.0007 (HIT) |

🎯 **No-LayerScale absorbed init holds at K=3475** — single seed each, both hit (tight). The absorption thesis works: LayerScale's main role is matching the projection's effective Frobenius norm, and a global uniform init at attn_scale=1.25 captures most of that benefit without LayerScale's per-layer per-channel learning.

## R52/R53/R54 — push K + scale variations under no-LayerScale

| run | K    | attn_scale | mlp_scale | final_val_loss | margin |
| --- | ---- | ---------- | --------- | -------------- | ------ |
| R51 | 3475 | 1.25       | 1.0       | **3.27958**    | +0.0004 (HIT) |
| R51 | 3500 | 1.25       | 1.0       | 3.27933        | +0.0007 (HIT) |
| R52 | 3450 | 1.25       | 1.0       | 3.28364        | -0.004 (miss) |
| R52 | 3425 | 1.25       | 1.0       | 3.28687        | -0.007 (miss) |
| R52 | 3400 | 1.25       | 1.0       | 3.28496        | -0.005 (miss) |
| R53 | 3475 | 1.5        | 1.0       | 3.28619        | -0.006 (miss) |
| R54 | 3475 | 2.0        | 1.0       | 3.28692        | -0.007 (miss) |
| R54 | 3450 | 2.0        | 1.0       | 3.28795        | -0.008 (miss) |

**Findings under no-LayerScale:**
- attn_scale=1.25 is the sweet spot (hits at K=3475, 3500). Higher scales (1.5, 2.0) miss — opposite of LayerScale-on (which preferred 2.5).
- K=3475 is the no-LS lower edge at attn_scale=1.25 (single seed). K≤3450 misses by 0.004+.
- LayerScale-on K=3475 best margin was 0.003 (scale=2.5); no-LS K=3475 best margin is 0.0004 — LayerScale-on is more robust at K=3475.

## R55-R58 results — K=3475 no-LS scale sweep is noise-bound

| run | K    | attn_scale | mlp_scale | final_val_loss | margin |
| --- | ---- | ---------- | --------- | -------------- | ------ |
| R51 | 3475 | 1.25       | 1.0       | **3.27958**    | +0.0004 (HIT) |
| R55 | 3475 | 1.25       | 1.0 (s2)  | 3.28159        | -0.0016 |
| R57 | 3475 | 1.0        | 1.0       | 3.28062        | -0.0006 |
| R57 | 3450 | 1.0        | 1.0       | 3.28248        | -0.0025 |
| R58 | 3475 | 1.1        | 1.0       | 3.28177        | -0.0018 |

K=3475 no-LS: scales {1.0, 1.1, 1.25} all within ~0.002 of target — single-seed margin is tight, only R51 hit cleanly. K=3450 misses by 0.002+. **The no-LS K=3475 hit is noise-flickering; need multi-seed and/or finer-grained init.**

## R59-R62 — larger attn_scale + vary mlp_scale at K=3475 (results)

| run | attn × mlp | final_val_loss | margin |
| --- | ---------- | -------------- | ------ |
| R59 | 2.5 × 1.0  | 3.29330        | -0.013 |
| R60 | 3.0 × 1.0  | 3.29813        | -0.018 |
| R61 | 1.25 × 1.5 | **3.27914**    | **+0.001 (HIT)** |
| R62 | 2.5 × 1.5  | 3.28429        | -0.004 |

K=3475 no-LS (attn × mlp) full grid:

| attn \ mlp | 1.0 | 1.5 |
| ---------- | --- | --- |
| 1.0 | 3.28062 (miss) | (R64) |
| 1.25 | 3.27958/3.28159/3.28241 (1/3 hit) | **3.27914 HIT** |
| 1.5 | 3.28619 (miss) | - |
| 2.0 | 3.28692 (miss) | - |
| 2.5 | 3.29330 (miss) | 3.28429 (miss) |
| 3.0 | 3.29813 (miss) | - |

mlp_scale=1.5 substantially helps across attn values (e.g. attn=2.5 went 3.29330 → 3.28429). Best so far: attn=1.25 mlp=1.5 single-seed HIT.

## R63+ — full per-module init × K-push under no-LayerScale (2026-04-28/29)

After per-module init flags landed, the dominant axis became `mlp_proj_init_scale` (mlp.proj output scale), with `mlp_fc_init_scale` (mlp.fc input scale) and matrix LR providing the second-order gains. Best configs at decreasing K under no-LayerScale hard constraint:

| K    | matrix_lr | attn_scale | mlp_scale | mlp_fc_scale | hits | best_val_loss |
| ---- | --------- | ---------- | --------- | ------------ | ---- | ------------- |
| 3475 | 0.018     | 1.25       | 3.0       | 1.0          | 1/1  | 3.27477       |
| 3450 | 0.018     | 1.25       | 1.5       | 1.0          | 1/1  | 3.27776       |
| 3425 | 0.018     | 1.25       | 3.0       | 1.0          | 1/1  | 3.27712 (early stop @ 3375) |
| 3400 | 0.018     | 1.25       | 1.5       | 1.5          | 1/1  | 3.27831 |
| 3375 | 0.018     | 1.25       | 1.5       | 1.5          | 1/1  | 3.27831 |
| 3350 | 0.018     | 1.25       | 1.5       | 1.5          | 2/2  | 3.27876 |
| 3325 | 0.018     | 1.25       | 2.5       | 1.5          | 2/2  | 3.27789 |
| 3300 | 0.018     | 1.25       | 3.0       | 1.5          | 2/2  | 3.27932 |
| 3275 | 0.019     | 1.25       | 3.0       | 1.5          | 2/2  | 3.27991 |
| **3250** | **0.019** | **1.25** | **3.0** | **2.0**     | **2/2** | **3.27934** |
| 3225 | 0.019     | 1.25       | 3.0       | 2.0          | 0/3+ | 3.28077 (closest miss) |
| 3200 | 0.019     | 1.25       | 3.0       | 2.0          | 0/1  | 3.28394 |

**Record under hard constraint: K=3250 reliably hits.** Saved logfile: `records/track_3_optimization/ed78d38d-513d-4b7b-8a61-c4a0d7f787ae.txt`. README row 3 updated.

200 steps below baseline-muon (3550), 200 steps below R2 record (3450).

K=3225 misses across all single-seed variants (attn=1.5, mom=0.93, mlp_fc=2.5, lr=0.018/0.019/0.020). Closest miss at lr=0.019 mlp_fc=2.0: 3.28077 (margin 0.0008). Multi-seed could swing.

### Things that did NOT help under no-LayerScale
- attn_scale > 1.5: hurts (3.28-3.29 at K=3475 with scale=2.0/2.5/3.0).
- qkv_init_scale=1.5: slight regression (3.28187 vs 3.27914 at K=3475).
- momentum=0.97: catastrophic (3.29 at K=3475).
- momentum=0.93: slight regression (3.28552 at K=3225 vs 3.28077 at mom=0.95).
- mlp_fc_init_scale=2.5 (vs 2.0): slight regression at K=3225.

## R63-R68 — multi-seed best + per-module init (earlier exploration)

- R63: attn=1.25, mlp=1.5 (seed 2, multi-seed best)
- R64: attn=1.0, mlp=1.5
- R65: attn=1.25, mlp=2.0
- R66: attn=1.25, mlp=1.5, **qkv=1.5** (per-module: try qkv axis)
- R67: attn=1.25, mlp=1.5, **mlp_fc=1.5** (per-module: try mlp.fc axis)
- R68: attn=1.25, mlp=1.5, K=3450 (push lower K at best)

6 configs sequential ~150 min.

## Code: per-module init scale (added 19:13 PDT)

Per user feedback ("fine-grained init per module and learning rate"): added `--qkv-init-scale` and `--mlp-fc-init-scale` flags to `train_gpt_h.py` (modal_sweep.py wires them through). Per-module LR multipliers (`--qkv-lr-mult`, `--mlp-fc-lr-mult`, `--attn-proj-lr-mult`, `--mlp-proj-lr-mult`) already exist.

Now have 4 init scales × 4 LR mults = 8 per-module knobs available. Next sweeps can vary qkv/mlp.fc/attn.proj/mlp.proj independently.

## Planned next (after R35 lands)
1. **If R35 hits 3.28 by step 3500** → save logfile to `records/track_3_optimization/<uuid>.txt`, update README row 2 to point at it (description: "MuonH + LayerScale (proj_scale=0 init) + Kaiming-proj, lr=0.017 hcd=1.0"), delete stale K=3450 logfile `cd2ed660-9a9d-4217-9917-805abe500a9c.txt`.
2. **If R35 misses** → run a second seed (single-seed noise ~0.0006 at this LR). If still miss, accept that the fix slightly changes numerics and tune lr ∈ {0.0165, 0.0175, 0.018} around the new operating point.
3. **Push below K=3500** under the same constraints — try K=3475 lr ∈ {0.017, 0.018, 0.019}, K=3450 lr ∈ {0.018, 0.020}, hcd=1.0. (Prior K<3500 sweep was at hcd=0.99 no-LS; the hard-constraint dynamics differ.)
4. Multi-seed confirmation of the best K under hard constraints (≥3 seeds for the publish row).

h100-8 capacity ~1 → runs sequential; ~25 min wall time per K=3500 run.
