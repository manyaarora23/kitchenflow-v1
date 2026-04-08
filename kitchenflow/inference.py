#!/usr/bin/env python3
"""
inference.py — KitchenFlow-v1 Ghost Kitchen Dispatcher Baseline
===============================================================
Runs an LLM agent against all 3 tasks. Each step = 1 simulation minute.

Required environment variables:
    API_BASE_URL   LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key

Usage:
    python inference.py
    python inference.py --url http://localhost:7860
    python inference.py --task T1_single_order_dispatch
"""

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.request
import urllib.error
from typing import Optional

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

TEMPERATURE = 0.1
MAX_TOKENS  = 256

SYSTEM_PROMPT = textwrap.dedent("""
    You are the AI dispatcher for a ghost kitchen delivery hub.
    Every minute you receive a snapshot of active orders and must decide
    whether to summon a driver for each order.

    Physics:
    - Driver speed = 30 km/h ÷ traffic_index  (in km per minute: 0.5 / traffic_index)
    - Driver ETA (minutes) = driver_dist_km ÷ (0.5 / traffic_index)
    - Food cools at 1.8°C/minute while waiting; perfect temp = 75°C
    - Once summoned, a driver cannot be un-summoned

    Strategy:
    - Summon driver when: food_ready_min - current_time ≈ driver_eta_min
    - i.e. dispatch so driver arrives just as food is bagged
    - food_ready_min ≈ current_time + (1 - food_prep_progress) × prep_time_min
    - (prep_time is NOT shown — infer it from food_prep_progress vs time)

    Reply with ONLY a valid JSON object:
    {"dispatch_decisions": {"ORD001": 0, "ORD002": 1}}
    Values: 0 = wait, 1 = summon driver now (ignored if already summoned)
    Include every active order ID in the response.
""").strip()


def build_prompt(obs: dict) -> str:
    lines = [
        f"MINUTE: {obs['time_min']} / {obs['max_time_min']}",
        f"TRAFFIC INDEX: {obs['traffic_index']} "
        f"(driver speed = {0.5 / obs['traffic_index']:.3f} km/min)",
        "",
        "ORDERS:",
    ]
    for o in obs["orders"]:
        if o["delivered"] or o["failed"]:
            lines.append(f"  {o['order_id']} [{o['status'].upper()}]")
            continue

        eta_str = f"ETA={o['driver_eta_min']}min" if o["driver_summoned"] else "not summoned"
        lines.append(
            f"  {o['order_id']} | {o['item_name']}"
            f" | prep={o['food_prep_progress']*100:.0f}%"
            f" | dist={o['driver_dist_km']:.2f}km ({eta_str})"
            f" | temp={o['food_temp_c']:.1f}°C"
            f" | food_ready={'YES' if o['food_ready'] else 'no'}"
            f" | driver_arrived={'YES' if o['driver_arrived'] else 'no'}"
        )
        if o["driver_arrived"] and not o["food_ready"]:
            lines.append(f"    ⚠️  DRIVER WAITING {o['minutes_driver_waited']}min — dispatch soon!")
        if o["food_ready"] and not o["driver_summoned"]:
            lines.append(f"    🔥 FOOD READY but NO driver summoned — food getting cold!")

    if obs.get("last_action_feedback") and obs.get("attempts", 0) > 1:
        lines += ["", f"LAST EVENT: {obs['last_action_feedback'][-200:]}"]

    lines += ["", "Your JSON decision (include all active order IDs):"]
    return "\n".join(lines)


def call_llm(client: OpenAI, prompt: str) -> dict:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = (completion.choices[0].message.content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"dispatch_decisions": {}}


# ── HTTP Client ───────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self._url = base_url.rstrip("/")
        self._episode_id: Optional[str] = None

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self._url}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self._url}{path}", timeout=10) as r:
            return json.loads(r.read())

    def reset(self, task_id: Optional[str] = None) -> dict:
        body: dict = {}
        if task_id:
            body["task_id"] = task_id
        obs = self._post("/reset", body)
        self._episode_id = obs.get("episode_id")
        return obs

    def step(self, action: dict) -> dict:
        body: dict = {"action": action}
        if self._episode_id:
            body["episode_id"] = self._episode_id
        return self._post("/step", body)

    def tasks(self) -> list:
        try:
            return [t["task_id"] for t in self._get("/tasks").get("tasks", [])]
        except Exception:
            return [
                "T1_single_order_dispatch",
                "T2_multi_order_coordination",
                "T3_peak_hour_rush",
            ]


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client: OpenAI, env: EnvClient, task_id: str) -> float:
    print(f"\n{'─'*62}")
    print(f"  Task : {task_id}")
    print(f"{'─'*62}")

    obs      = env.reset(task_id=task_id)
    max_mins = obs.get("max_time_min", 30)
    n_orders = len(obs.get("orders", []))

    print(f"  Orders: {n_orders} | Time limit: {max_mins} min")
    for o in obs.get("orders", []):
        print(f"    {o['order_id']} {o['item_name']} — "
              f"driver {o['driver_dist_km']}km away")

    best_score = 0.0

    for minute in range(1, max_mins + 1):
        if obs.get("done", False):
            break

        prompt = build_prompt(obs)
        try:
            action_data = call_llm(client, prompt)
        except Exception as exc:
            print(f"  [min{minute}] LLM error: {exc}")
            action_data = {"dispatch_decisions": {}}

        try:
            obs = env.step(action_data)
        except RuntimeError as exc:
            print(f"  [min{minute}] Step error: {exc}")
            break

        score   = obs.get("score", 0.0)
        reward  = obs.get("reward", 0.0)
        best_score = max(best_score, score)

        # Print only interesting minutes
        fb = obs.get("last_action_feedback", "")
        dispatches = {k: v for k, v in
                      action_data.get("dispatch_decisions", {}).items() if v == 1}
        if dispatches or any(k in fb for k in ["READY", "ARRIVED", "DELIVERED", "FAILED"]):
            icon = "✅" if score >= 0.9 else ("🟡" if score > 0.4 else "")
            print(f"  min{minute:02d}: {fb[:100]}  {icon}")

        if obs.get("done", False):
            final = obs.get("score", 0.0)
            best_score = max(best_score, final)
            print(f"\n  ── Final score: {final:.2f} ──")
            break

    return best_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KitchenFlow-v1 baseline agent")
    parser.add_argument("--url",  default=ENV_URL)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    print("=" * 62)
    print("  KitchenFlow-v1 — Ghost Kitchen Dispatcher Baseline")
    print("=" * 62)
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Env URL: {args.url}")
    print()
    print("  Baseline comparison (80% threshold heuristic):")
    print("    T1 easy   → 1.00")
    print("    T2 medium → 0.96")
    print("    T3 hard   → 0.84")
    print("    Average   → 0.93")

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(args.url)

    task_ids = [args.task] if args.task else env.tasks()
    scores   = {}
    start    = time.time()

    for tid in task_ids:
        try:
            scores[tid] = run_task(llm, env, tid)
        except Exception as exc:
            print(f"\n  ERROR on {tid}: {exc}")
            scores[tid] = 0.0

    elapsed = time.time() - start
    avg     = sum(scores.values()) / len(scores) if scores else 0.0

    print(f"\n{'='*62}")
    print("  RESULTS")
    print(f"{'='*62}")
    diff_map = {
        "T1_single_order_dispatch":    "easy",
        "T2_multi_order_coordination": "medium",
        "T3_peak_hour_rush":           "hard",
    }
    for tid, s in scores.items():
        bar  = "█" * int(s * 20)
        diff = diff_map.get(tid, "")
        print(f"  {tid:<35} {s:.2f}  [{diff}]")
        print(f"    |{bar:<20}|")
    print(f"  {'─'*55}")
    print(f"  {'AVERAGE':<35} {avg:.2f}  (baseline: 0.93)")
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"{'='*62}")
    sys.exit(0 if avg > 0 else 1)


if __name__ == "__main__":
    main()
