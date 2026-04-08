"""
KitchenFlow-v1 — Ghost Kitchen Dispatcher
Data models for the OpenEnv environment.

The agent acts as the "Brain" of a delivery hub, deciding WHEN to summon
a driver for each order based on real-time kitchen and traffic data.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class KitchenAction(Action):
    """
    Per-step dispatch decision.

    For each active order, submit 0 (wait) or 1 (summon driver).
    Once a driver is summoned for an order, further '1' actions are ignored.

    Example:
        {"dispatch_decisions": {"ORD001": 1, "ORD002": 0, "ORD003": 1}}
    """
    dispatch_decisions: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Map of order_id → action. "
            "0 = wait this minute, 1 = summon the driver now. "
            "Omitted orders default to 0 (wait)."
        )
    )


class KitchenObservation(Observation):
    """
    One-minute snapshot of the ghost kitchen hub.

    Contains real-time data for every active order plus the current
    traffic index — everything the agent needs to time its dispatch.
    """
    # Task context
    task_id: str = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="What the agent must do")
    difficulty: str = Field(default="", description="easy / medium / hard")

    # Simulation clock
    time_min: int = Field(default=0, description="Current simulation minute")
    max_time_min: int = Field(default=30, description="Episode time limit (minutes)")

    # City-level traffic (Uber Movement-style congestion index)
    traffic_index: float = Field(
        default=1.0,
        description=(
            "Road congestion multiplier (1.0 = free flow, 2.5 = gridlock). "
            "Higher traffic → drivers travel slower → longer ETAs."
        )
    )

    # Per-order states
    orders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of active order states. Each dict contains: "
            "order_id, item_name, food_prep_progress (0–1), "
            "driver_dist_km, food_temp_c, driver_summoned (bool), "
            "driver_eta_min (None or int), food_ready (bool), "
            "driver_arrived (bool), delivered (bool), status, "
            "minutes_food_waited, minutes_driver_waited."
        )
    )

    # Episode-level tracking
    orders_delivered: int = Field(default=0)
    orders_failed: int = Field(default=0)
    total_temp_penalty: float = Field(default=0.0)
    total_waste_penalty: float = Field(default=0.0)

    # Feedback
    last_action_feedback: str = Field(default="", description="What happened last minute")
    score: float = Field(default=0.0, description="Running score 0.0–1.0")
    attempts: int = Field(default=0, description="Steps (minutes) elapsed")
    max_attempts: int = Field(default=30, description="Max steps")
