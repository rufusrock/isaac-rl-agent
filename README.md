# isaac-rl-agent

A vision-based reinforcement learning agent for *The Binding of Isaac: Afterbirth+*.
Pipeline: screen-capture → behavior cloning (BC) → PPO fine-tuning, with rewards
derived from a Lua mod that streams authoritative game state over UDP.

---

## Status — 2026-04-14

**Phase 1 — Behavior Cloning (done, iterating):** best checkpoint reaches
`movement_acc = 0.820`, `shooting_acc = 0.969` on a held-out run (4 rollout
runs, ~132k frames, 10k held-out).

**Phase 2 — Architecture & hyperparameter sweep (currently running):**
`sweep_v2` is a phased sweep with carry-forward: each phase fixes every axis
but one, picks the winner by held-out movement accuracy, and carries it into
the next phase. Phases:

| Phase | Axis | Variants |
|-------|------|----------|
| A    | Architecture | `nature`, `deep4`, `impala_small`, `impala_med`, `plain_deeper`, `plain_wider` |
| A+   | Impala scaling (BBF / Procgen inspired) | `impala_wide`, `impala_xl`, `impala_4stage`, `impala_3rb` |
| B    | Input resolution | 84, 128, 196, 256 |
| C    | Frame stack | 1, 2, 4, 6, 8 |
| D    | Colour / motion | `gray`, `rgb`, `mc_sat`, `gray+diff`, `rgb+diff` |
| E    | Normalization | `batch`, `layer`, `group`, `none` |
| F    | Augmentation | `flip`, `flip+drq`, `flip+jitter`, `none` |

A-phase winner so far is `A_impala_med` at `holdout_movement_acc = 0.640`.

**Phase 3 — RL fine-tuning:** upcoming. The best BC checkpoint will be used
to warm-start a PPO loop against the live telemetry environment.

---

## Collaboration

This project is being built in pair-programming style with
**Claude (Anthropic's AI assistant)**. I (the human) drive the research
direction and experiment design; Claude helps with implementation, debugging,
code review, and keeping the code tidy. Credit is shared — this isn't a
solo-human project and I don't want to pretend it is.

---

## Repo layout

```
binding_rl_agent/       core library
  window_capture.py     grab client-area frames from the Isaac window
  env.py                frame-stacking observation wrapper
  preprocessing.py      resize / grayscale / multi-channel transforms
  dataset.py            BC dataset, memmap cache, uint8 transfer
  models.py             CNN policies (nature, plain deep, Impala)
  training.py           BC training loop
  game_state.py         UDP telemetry receiver
  reward_detection.py   telemetry-driven reward/done signals
  rl_env.py             vision-first RL env with reward scaffolding
  rl_training.py        actor-critic / PPO loop
  room_graph.py         room adjacency + nav hints
  inspection.py         rollout contact sheets, GIF previews
  input_controller.py   synthetic key input
  recording.py          manual rollout recorder
  inference.py          checkpoint loading + live prediction

isaac_mod/
  ScoreStreamer/        Afterbirth+ Lua mod — streams game state to 127.0.0.1:8123

train_bc.py             entry point for behavior cloning
train_rl.py             entry point for online RL
manual_rollout.py       record (observation, action) pairs while you play
live_capture.py         preview the capture stream
live_inference.py       live policy predictions (no control)
live_policy_control.py  safety-gated live control loop (F8 arm, F9 e-stop)
debug_input_control.py  verify synthetic keystrokes reach Isaac
debug_reward_signals.py preview live telemetry rewards
inspect_rollout.py      contact sheet + GIF of a recorded rollout
inspect_rl_run.py       summarise a saved RL run
plot_recent_rl_runs.py  plot training curves
```

Locally-kept-but-not-pushed: `rollouts/` (~47 GB), `models/`,
`rl_runs/`, `artifacts/`, sweep scripts and CSVs.

---

## Game-side setup

Install the `isaac_mod/ScoreStreamer` folder into:

```
%USERPROFILE%\Documents\My Games\Binding of Isaac Afterbirth+ Mods\
```

Enable it in the in-game mods menu. With the mod running, Isaac emits UDP
packets to `127.0.0.1:8123` carrying room state, kills, damage, deaths, and
floor-map info — consumed by `binding_rl_agent.game_state`.

---

## Quick tour

Record a rollout while you play:

```powershell
python manual_rollout.py --steps 500 --fps 10
```

Train a BC policy on everything under `rollouts/`:

```powershell
python train_bc.py --epochs 15
```

Preview predictions on the live game (no control):

```powershell
python live_inference.py
```

Safety-gated live control (starts disarmed; `F8` to arm, `F9` e-stop, `q` to quit):

```powershell
python live_policy_control.py
```

Online RL from the latest BC checkpoint:

```powershell
python train_rl.py --updates 100 --rollout-steps 128
```

---

## Lessons learned (so far)

- **Val split must be random, not temporal.** Early experiments used a
  within-run first-80% / last-20% split, which created an early-game →
  late-game distribution shift and made val loss diverge from train loss.
  Random per-sample split fixed this.
- **GPU starvation on Windows is real.** Moving to `uint8` tensors on the
  DataLoader side plus a memmap preprocessed cache kept the 3070 fed.
- **VRAM overflow is silent on Windows.** Large Impala variants at
  `batch=256` quietly spilled into shared system memory via PCIe, collapsing
  throughput ~100× while still reporting 95% GPU util. `batch=128` is safe.

---

## Requirements

Python 3.10+, a GPU is strongly recommended. Dependencies in `requirements.txt`.

```powershell
pip install -r requirements.txt
```
