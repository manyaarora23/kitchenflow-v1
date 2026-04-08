"""
KitchenFlow-v1 — Ghost Kitchen Dispatcher Environment

Three tasks of increasing difficulty:
  T1  single_order_dispatch    Easy   — 1 order, stable traffic, 30-min window
  T2  multi_order_coordination Medium — 3 orders, varying preps & distances
  T3  peak_hour_rush           Hard   — 5 orders, traffic spikes, dynamic conditions

Simulation mechanics (each step = 1 minute):
  • Food prep advances linearly to 1.0 over its prep_time_min
  • Once food is ready (progress=1.0), it starts cooling at COOLING_RATE °C/min
  • When driver is summoned, ETA = dist_km / (BASE_SPEED_KM_MIN / traffic_index)
  • Driver distance shrinks each minute at their current travel speed
  • Delivery happens when: food_ready AND driver_arrived
  • Episode ends when all orders resolved OR time_limit reached

Reward function (per order):
  +10  if |driver_arrival_min - food_ready_min| ≤ 2   (perfect timing)
  +5   if |driver_arrival_min - food_ready_min| ≤ 5   (good timing)
  -1   per °C below PERFECT_SERVING_TEMP at delivery
  -20  if driver waits > 15 min (risk of cancellation)
  -5   if food waits > 10 min  (cold food + customer dissatisfaction)
  -10  if order not resolved within time limit (failed delivery)
"""

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    _has_openenv = True
except ImportError:
    _has_openenv = False
    # Minimal stubs so the server still runs without openenv installed
    class State:
        def __init__(self, episode_id=None, step_count=0):
            self.episode_id = episode_id
            self.step_count = step_count

    class Environment:
        pass

try:
    from ..models import KitchenAction, KitchenObservation
except ImportError:
    from models import KitchenAction, KitchenObservation

# ─────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────
BASE_SPEED_KM_MIN   = 0.50   # 30 km/h city average
INITIAL_FOOD_TEMP   = 85.0   # °C when food is freshly prepared
PERFECT_TEMP        = 75.0   # °C ideal serving temperature
COOLING_RATE        = 1.8    # °C per minute food sits waiting
DELIVERY_COOL_RATE  = 0.4    # °C per minute during transit (insulated bag)
MAX_DRIVER_WAIT_MIN = 15     # minutes — driver may cancel after this
MAX_FOOD_WAIT_MIN   = 10     # minutes — food quality degrades badly

# ─────────────────────────────────────────────
# Traffic patterns  (Uber Movement-style)
# ─────────────────────────────────────────────
def _traffic_at(minute: int, scenario: str) -> float:
    """Return traffic index for a given simulation minute and scenario."""
    if scenario == "stable":
        return 1.2

    elif scenario == "moderate":
        base = 1.3
        wave = 0.3 * math.sin(math.pi * minute / 20)
        return round(base + wave, 2)

    elif scenario == "peak_hour":
        if minute < 8:
            return 1.1
        elif minute < 12:
            return 1.1 + (minute - 8) * 0.25
        elif minute < 20:
            return 2.1 + 0.1 * math.sin(minute)
        else:
            return 2.1 - (minute - 20) * 0.08
    return 1.0


# ─────────────────────────────────────────────
# Order template definitions
# ─────────────────────────────────────────────
ORDER_TEMPLATES = {
    "fast":    {"item_name": "Fries & Shake",      "prep_time_min": 10, "driver_dist_km": 1.2},
    "medium":  {"item_name": "Chicken Burger",     "prep_time_min": 15, "driver_dist_km": 2.0},
    "slow":    {"item_name": "Loaded Nachos",      "prep_time_min": 20, "driver_dist_km": 3.5},
    "complex": {"item_name": "Full Grill Platter", "prep_time_min": 25, "driver_dist_km": 4.0},
    "express": {"item_name": "Hot Dog Combo",      "prep_time_min":  8, "driver_dist_km": 0.8},
}


def _make_order(order_id: str, template_key: str) -> Dict[str, Any]:
    t = ORDER_TEMPLATES[template_key]
    return {
        "order_id":              order_id,
        "item_name":             t["item_name"],
        "template":              template_key,
        "prep_time_min":         t["prep_time_min"],
        "food_prep_progress":    0.0,
        "driver_dist_km":        t["driver_dist_km"],
        "_initial_dist_km":      t["driver_dist_km"],
        "food_temp_c":           INITIAL_FOOD_TEMP,
        "driver_summoned":       False,
        "driver_summon_min":     None,
        "driver_eta_min":        None,
        "driver_arrived":        False,
        "driver_arrived_min":    None,
        "food_ready":            False,
        "food_ready_min":        None,
        "delivered":             False,
        "failed":                False,
        "minutes_food_waited":   0,
        "minutes_driver_waited": 0,
        "status":                "preparing",
        "score":                 None,
    }


# ─────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────
SCENARIOS = {
    "T1_single_order_dispatch": {
        "description": (
            "One order is being prepared. Your job is to summon the driver "
            "at exactly the right moment so they arrive just as the food is bagged. "
            "Watch food_prep_progress (0→1) and driver_dist_km. "
            "Traffic is stable. Submit dispatch_decisions: {ORD001: 1} to call the driver."
        ),
        "difficulty":    "easy",
        "max_time_min":  30,
        "traffic_mode":  "stable",
        "orders":        [("ORD001", "medium")],
    },
    "T2_multi_order_coordination": {
        "description": (
            "Three orders are in progress — a quick snack, a burger, and a loaded platter. "
            "Each has a different prep time and driver distance. "
            "Coordinate dispatch so all three drivers arrive on time. "
            "Traffic fluctuates — watch the traffic_index each minute."
        ),
        "difficulty":    "medium",
        "max_time_min":  35,
        "traffic_mode":  "moderate",
        "orders":        [("ORD001", "fast"), ("ORD002", "medium"), ("ORD003", "slow")],
    },
    "T3_peak_hour_rush": {
        "description": (
            "Five orders land during the lunch rush. "
            "Traffic spikes between minutes 10–20 (index up to 2.2), "
            "slowing all drivers. One driver may be far from the hub. "
            "Maximise food quality and minimise driver idle time across all 5 orders."
        ),
        "difficulty":    "hard",
        "max_time_min":  45,
        "traffic_mode":  "peak_hour",
        "orders":        [
            ("ORD001", "express"),
            ("ORD002", "fast"),
            ("ORD003", "medium"),
            ("ORD004", "slow"),
            ("ORD005", "complex"),
        ],
    },
}


# ─────────────────────────────────────────────
# Per-order grader
# ─────────────────────────────────────────────
def _score_order(order: Dict[str, Any]) -> Tuple[float, str]:
    """Grade a single completed or failed order. Returns (score 0–1, feedback)."""
    if order["failed"]:
        return 0.0, f"{order['order_id']} FAILED (timeout) → score=0.0"

    food_ready_min     = order["food_ready_min"]    or 0
    driver_arrived_min = order["driver_arrived_min"] or 0
    food_waited        = order["minutes_food_waited"]
    driver_waited      = order["minutes_driver_waited"]
    temp               = order["food_temp_c"]

    raw = 0.0
    timing_gap = abs(driver_arrived_min - food_ready_min)
    if timing_gap <= 2:
        raw += 10
    elif timing_gap <= 5:
        raw += 5

    temp_penalty = max(0.0, PERFECT_TEMP - temp)
    raw -= temp_penalty

    if driver_waited > MAX_DRIVER_WAIT_MIN:
        raw -= 20
    if food_waited > MAX_FOOD_WAIT_MIN:
        raw -= 5

    score = max(0.0, min(1.0, (raw + 35) / 45.0))

    fb = (
        f"{order['order_id']} ({order['item_name']}): "
        f"food_ready=min{food_ready_min} driver=min{driver_arrived_min} "
        f"gap={timing_gap}min temp={temp:.1f}°C "
        f"raw={raw:.1f} score={score:.2f}"
    )
    return round(score, 4), fb


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
class KitchenflowEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state    = State(episode_id=str(uuid4()), step_count=0)
        self._task_id  = ""
        self._task_idx = 0
        self._scenario: Dict[str, Any] = {}
        self._orders:   List[Dict[str, Any]] = []
        self._time_min  = 0
        self._max_time  = 30
        self._traffic_mode = "stable"
        self._done     = False
        self._episode_score = 0.0
        self._order_scores: Dict[str, float] = {}

    def _traffic(self) -> float:
        return _traffic_at(self._time_min, self._traffic_mode)

    def _order_snapshot(self, o: Dict) -> Dict[str, Any]:
        return {
            "order_id":              o["order_id"],
            "item_name":             o["item_name"],
            "food_prep_progress":    round(o["food_prep_progress"], 3),
            "driver_dist_km":        round(o["driver_dist_km"], 2),
            "food_temp_c":           round(o["food_temp_c"], 1),
            "driver_summoned":       o["driver_summoned"],
            "driver_eta_min":        o["driver_eta_min"],
            "food_ready":            o["food_ready"],
            "driver_arrived":        o["driver_arrived"],
            "delivered":             o["delivered"],
            "failed":                o["failed"],
            "minutes_food_waited":   o["minutes_food_waited"],
            "minutes_driver_waited": o["minutes_driver_waited"],
            "status":                o["status"],
        }

    def _build_obs(self, feedback: str) -> KitchenObservation:
        delivered = sum(1 for o in self._orders if o["delivered"])
        failed    = sum(1 for o in self._orders if o["failed"])
        t_penalty = sum(
            max(0, PERFECT_TEMP - o["food_temp_c"])
            for o in self._orders if o["delivered"]
        )
        w_penalty = sum(
            20 for o in self._orders
            if o["delivered"] and o["minutes_driver_waited"] > MAX_DRIVER_WAIT_MIN
        )
        running_score = (
            sum(self._order_scores.values()) / len(self._orders)
            if self._order_scores else 0.0
        )
        return KitchenObservation(
            task_id=self._task_id,
            task_description=self._scenario.get("description", ""),
            difficulty=self._scenario.get("difficulty", ""),
            time_min=self._time_min,
            max_time_min=self._max_time,
            traffic_index=round(self._traffic(), 2),
            orders=[self._order_snapshot(o) for o in self._orders],
            orders_delivered=delivered,
            orders_failed=failed,
            total_temp_penalty=round(t_penalty, 1),
            total_waste_penalty=round(w_penalty, 1),
            last_action_feedback=feedback,
            score=round(running_score, 4),
            attempts=self._time_min,
            max_attempts=self._max_time,
            done=self._done,
            reward=0.0,
        )

    def reset(self, task_id: Optional[str] = None) -> KitchenObservation:
        task_ids = list(SCENARIOS.keys())
        if task_id and task_id in SCENARIOS:
            self._task_id = task_id
        else:
            self._task_id = task_ids[self._task_idx % len(task_ids)]
            self._task_idx += 1

        self._scenario     = SCENARIOS[self._task_id]
        self._time_min     = 0
        self._max_time     = self._scenario["max_time_min"]
        self._traffic_mode = self._scenario["traffic_mode"]
        self._done         = False
        self._episode_score = 0.0
        self._order_scores  = {}
        self._state         = State(episode_id=str(uuid4()), step_count=0)

        self._orders = [
            _make_order(oid, tkey)
            for oid, tkey in self._scenario["orders"]
        ]

        return self._build_obs(
            "Kitchen is open. Watch food_prep_progress and traffic_index carefully. "
            "Summon the driver at the right moment!"
        )

    def step(self, action: KitchenAction) -> KitchenObservation:
        if not self._orders:
            self.reset()

        self._state.step_count += 1
        self._time_min += 1
        traffic = self._traffic()
        speed   = BASE_SPEED_KM_MIN / traffic

        events: List[str] = [f"Min {self._time_min} | traffic={traffic:.2f}"]

        for o in self._orders:
            if o["delivered"] or o["failed"]:
                continue

            oid = o["order_id"]

            if not o["food_ready"]:
                o["food_prep_progress"] = min(
                    1.0,
                    o["food_prep_progress"] + 1.0 / o["prep_time_min"]
                )
                if o["food_prep_progress"] >= 1.0:
                    o["food_ready"]     = True
                    o["food_ready_min"] = self._time_min
                    o["status"]         = "food_ready"
                    o["food_temp_c"]    = INITIAL_FOOD_TEMP
                    events.append(f"  🍔 {oid} food READY (min {self._time_min})")

            decision = action.dispatch_decisions.get(oid, 0)
            if decision == 1 and not o["driver_summoned"] and not o["delivered"]:
                o["driver_summoned"]  = True
                o["driver_summon_min"] = self._time_min
                eta = o["driver_dist_km"] / speed
                o["driver_eta_min"] = round(eta, 1)
                o["status"] = "driver_en_route"
                events.append(
                    f"  🛵 {oid} driver SUMMONED — dist={o['driver_dist_km']:.1f}km "
                    f"ETA≈{eta:.1f}min"
                )

            if o["driver_summoned"] and not o["driver_arrived"]:
                moved = speed
                o["driver_dist_km"] = max(0.0, o["driver_dist_km"] - moved)
                if speed > 0:
                    o["driver_eta_min"] = round(o["driver_dist_km"] / speed, 1)

                if o["driver_dist_km"] <= 0.0:
                    o["driver_arrived"]     = True
                    o["driver_arrived_min"] = self._time_min
                    o["driver_dist_km"]     = 0.0
                    o["driver_eta_min"]     = 0
                    events.append(f"  🏍️  {oid} driver ARRIVED (min {self._time_min})")

            if o["food_ready"] and not o["delivered"]:
                if not o["driver_arrived"]:
                    o["food_temp_c"] = max(40.0, o["food_temp_c"] - COOLING_RATE)
                    o["minutes_food_waited"] += 1
                else:
                    o["food_temp_c"] = max(40.0, o["food_temp_c"] - 0.5)

            if o["driver_arrived"] and not o["food_ready"]:
                o["minutes_driver_waited"] += 1

            if o["food_ready"] and o["driver_arrived"] and not o["delivered"]:
                o["delivered"] = True
                o["status"]    = "delivered"
                s, fb = _score_order(o)
                o["score"]     = s
                self._order_scores[oid] = s
                events.append(f"  ✅ {oid} DELIVERED  {fb}")

            if self._time_min >= self._max_time and not o["delivered"]:
                o["failed"] = True
                o["status"] = "failed"
                self._order_scores[oid] = 0.0
                events.append(f"  ❌ {oid} FAILED (timeout)")

        all_resolved = all(o["delivered"] or o["failed"] for o in self._orders)
        self._done   = all_resolved or self._time_min >= self._max_time

        if self._done and self._order_scores:
            self._episode_score = round(
                sum(self._order_scores.values()) / len(self._orders), 4
            )

        step_reward = self._shaped_reward()

        feedback = " | ".join(events)
        obs = self._build_obs(feedback)
        obs.score  = self._episode_score if self._done else round(
            sum(self._order_scores.values()) / len(self._orders), 4
        ) if self._order_scores else 0.0
        obs.reward = round(step_reward, 4)
        return obs

    def _shaped_reward(self) -> float:
        reward = 0.0
        for o in self._orders:
            if o["delivered"]:
                continue
            if o["failed"]:
                reward -= 0.5
                continue
            if o["food_ready"] and o["food_temp_c"] >= PERFECT_TEMP:
                reward += 0.05
            if o["food_ready"] and not o["driver_summoned"]:
                reward -= 0.10
            if o["driver_arrived"] and not o["food_ready"]:
                reward -= 0.15
        return reward

    @property
    def state(self) -> State:
        return self._state

    def close(self):
        pass


# Expose task list for the /tasks endpoint
TASKS = [
    {
        "task_id":      tid,
        "description":  sc["description"],
        "difficulty":   sc["difficulty"],
        "max_time_min": sc["max_time_min"],
        "n_orders":     len(sc["orders"]),
    }
    for tid, sc in SCENARIOS.items()
]
