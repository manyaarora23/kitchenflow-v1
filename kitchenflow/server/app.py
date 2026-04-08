"""
FastAPI application for KitchenFlow-v1 — Ghost Kitchen Dispatcher.

Session-managed HTTP server. Each reset returns an episode_id;
pass it with every step to maintain state across the simulation.

Endpoints:
    GET  /         → redirects to /docs (Swagger UI)
    POST /reset    Start a new episode
    POST /step     Advance simulation by 1 minute with dispatch decisions
    GET  /state    Current session state
    GET  /schema   Action / observation schemas
    GET  /metadata Environment metadata
    GET  /health   Health check
    POST /mcp      JSON-RPC 2.0 stub
    GET  /tasks    List all tasks
"""

import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

try:
    from .models import KitchenAction, KitchenObservation
    from .kitchenflow_env_environment import KitchenflowEnvironment, TASKS
except (ModuleNotFoundError, ImportError):
    from models import KitchenAction, KitchenObservation
    from kitchenflow_env_environment import KitchenflowEnvironment, TASKS

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: Dict[str, KitchenflowEnvironment] = {}
_lock = threading.Lock()
_DEFAULT = "default"


def _get_or_create(sid: str) -> KitchenflowEnvironment:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = KitchenflowEnvironment()
        return _sessions[sid]


def _obs_dict(obs: KitchenObservation, sid: str) -> dict:
    d = obs.model_dump()
    d["episode_id"] = sid
    return d


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KitchenFlow-v1 — Ghost Kitchen Dispatcher",
    version="1.0.0",
    description=(
        "OpenEnv environment where an AI agent dispatches delivery drivers "
        "for a ghost kitchen, timing each summon to minimise wait time, "
        "cold food, and driver cancellations.\n\n"
        "**Quick start:** Use `POST /reset` to start an episode, then `POST /step` to act."
    ),
)


# ── Root redirect → Swagger UI ────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Redirect root URL to Swagger UI automatically."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "kitchenflow_env", "tasks": len(TASKS)}


@app.get("/metadata")
def metadata():
    return {
        "name": "kitchenflow_env",
        "description": (
            "KitchenFlow-v1: Ghost Kitchen Dispatcher. "
            "An AI agent monitors food prep progress and real-time traffic "
            "to decide the perfect moment to summon each delivery driver — "
            "balancing food temperature, driver idle time, and delivery efficiency."
        ),
        "version": "1.0.0",
        "tasks": [t["task_id"] for t in TASKS],
    }


@app.get("/schema")
def schema():
    return {
        "action":      KitchenAction.model_json_schema(),
        "observation": KitchenObservation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
            },
        },
    }


@app.post("/reset")
def reset(body: Dict[str, Any] = Body(default={})):
    """
    Start a new episode.
    Optional: {"task_id": "T1_single_order_dispatch", "episode_id": "my-session"}
    """
    task_id = body.get("task_id")
    sid     = body.get("episode_id") or str(uuid.uuid4())
    env     = _get_or_create(sid)
    obs     = env.reset(task_id=task_id)
    return _obs_dict(obs, sid)


@app.post("/step")
def step(body: Dict[str, Any] = Body(...)):
    """
    Advance simulation by 1 minute.

    Body:
      action     (dict, required) — {"dispatch_decisions": {"ORD001": 1, "ORD002": 0}}
      episode_id (str, optional)  — session ID from reset

    Action values:
      0 = wait this minute
      1 = summon driver for this order (one-time trigger)
    """
    action_data = body.get("action")
    if action_data is None:
        raise HTTPException(status_code=422, detail="'action' field required")

    sid = body.get("episode_id", _DEFAULT)

    try:
        action = KitchenAction(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    env = _get_or_create(sid)
    if not env._orders:
        env.reset()

    obs = env.step(action)
    return _obs_dict(obs, sid)


@app.get("/state")
def state(episode_id: str = _DEFAULT):
    env = _get_or_create(episode_id)
    s   = env.state
    return {"episode_id": s.episode_id or episode_id, "step_count": s.step_count}


@app.post("/mcp")
def mcp(body: Dict[str, Any] = Body(default={})):
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {
            "name": "kitchenflow_env",
            "description": "KitchenFlow-v1 Ghost Kitchen Dispatcher OpenEnv environment",
        },
    })


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point — enables: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
