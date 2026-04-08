---
title: KitchenFlow-v1 Ghost Kitchen Dispatcher
emoji: 🍔
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# KitchenFlow-v1 — Ghost Kitchen Dispatcher

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent acts as the dispatch brain of a ghost kitchen — deciding the **perfect moment to summon a delivery driver** so they arrive exactly when the food is bagged.

> *Everyone knows the pain of cold fries. Now you can train an AI to prevent it.*

## The Problem

In busy cloud kitchens, food often sits getting cold because the driver isn't there yet — or drivers idle for 20 minutes because food isn't ready. The agent must watch food prep progress and real-time traffic to time each dispatch perfectly.

## Observation Space

Each step (= 1 minute) the agent sees:

| Field | Type | Description |
|-------|------|-------------|
| `time_min` | int | Current simulation minute |
| `traffic_index` | float | Road congestion (1.0 = free flow, 2.5 = gridlock) |
| `orders[].food_prep_progress` | 0.0–1.0 | How close the food is to being bagged |
| `orders[].driver_dist_km` | float | Distance of the assigned driver |
| `orders[].food_temp_c` | float | Current food temperature (°C) |
| `orders[].driver_eta_min` | int/null | Minutes until driver arrives (null if not summoned) |
| `orders[].driver_summoned` | bool | Has the driver been called? |

## Action Space

```json
{"dispatch_decisions": {"ORD001": 1, "ORD002": 0, "ORD003": 1}}
```
- `0` = Wait this minute  
- `1` = Summon driver (one-time trigger — driver heads to hub immediately)

## Reward Function

| Event | Reward |
|-------|--------|
| Driver arrives ≤ 2 min of food being ready | **+10** (perfect timing) |
| Driver arrives ≤ 5 min of food being ready | **+5** (good timing) |
| Each °C below 75°C at delivery | **−1** (cold food) |
| Driver waits > 15 min (risk of cancellation) | **−20** |
| Food waits > 10 min without driver | **−5** |
| Order not resolved in time | **−10** (failed) |

Scores are normalised to **0.0–1.0** per order, then averaged.

## Tasks

| Task ID | Difficulty | Scenario | Orders | Time Limit |
|---------|-----------|----------|--------|-----------|
| `T1_single_order_dispatch` | Easy | 1 burger, stable traffic (1.2) | 1 | 30 min |
| `T2_multi_order_coordination` | Medium | 3 orders, fluctuating traffic | 3 | 35 min |
| `T3_peak_hour_rush` | Hard | 5 orders, traffic spike 10–20min (up to 2.2) | 5 | 45 min |

## Dispatch Physics

```
driver_speed = 0.5 km/min ÷ traffic_index   (= 30 km/h in free flow)
driver_eta   = driver_dist_km ÷ driver_speed

optimal_dispatch_minute = food_ready_minute − driver_eta
```

## Baseline Scores (80% Threshold Heuristic)

*"Summon driver when food_prep_progress ≥ 0.80"*

| Task | Score |
|------|-------|
| T1 easy | 1.00 |
| T2 medium | 0.96 |
| T3 hard | 0.84 |
| **Average** | **0.93** |

A well-tuned LLM that reasons about ETA calculations should **beat 0.93**.

## API

```bash
# Start T1
curl -X POST https://your-space.hf.space/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "T1_single_order_dispatch"}'

# Step (dispatch decisions, with episode_id from reset)
curl -X POST https://your-space.hf.space/step \
  -H 'Content-Type: application/json' \
  -d '{"action": {"dispatch_decisions": {"ORD001": 1}}, "episode_id": "<from_reset>"}'
```

## Run Baseline

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=https://your-space.hf.space
python inference.py
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
# Visit http://localhost:7860 → auto-redirects to Swagger UI
```

## OpenEnv Validation

```bash
openenv validate .
# [OK] : Ready for multi-mode deployment
```
