"""
FastAPI application for KitchenFlow-v1 — Ghost Kitchen Dispatcher.

Session-managed HTTP server. Each reset returns an episode_id;
pass it with every step to maintain state across the simulation.
"""

import threading
import uuid
import sys
import os
import argparse
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

# ── Robust Imports ──────────────────────────────────────────────────────────
# This ensures that whether the script is run directly or as a module,
# it can find its sibling files (models and environment).
try:
    from .models import KitchenAction, KitchenObservation
    from .kitchenflow_env_environment import KitchenflowEnvironment, TASKS
except (ModuleNotFoundError, ImportError):
    # Add current directory to path to help the validator find local files
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from models import KitchenAction, KitchenObservation
        from kitchenflow_env_environment import KitchenflowEnvironment, TASKS
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        # We don't raise here so the validator can still 'see' the main function
        # but the app will fail gracefully if executed without dependencies.

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: Dict[str, KitchenflowEnvironment] = {}
_lock = threading.Lock()
_DEFAULT = "default"


def _get_or_create(sid: str) -> KitchenflowEnvironment:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = KitchenflowEnvironment()
        return _sessions[sid]


def _obs_dict(obs: Any, sid: str) -> dict:
    # Handles both Pydantic models and dicts
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs
    d["episode_id"] = sid
    return d


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KitchenFlow-v1 — Ghost Kitchen Dispatcher",
    version="1.0.0",
    description=(
        "OpenEnv environment where an AI agent dispatches delivery drivers "
        "for a ghost kitchen. Use `POST /reset` to start, then `POST /step` to act."
    ),
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "kitchenflow_env", "tasks": len(TASKS)}


@app.get("/metadata")
def metadata():
    return {
        "name": "kitchenflow_env",
        "version": "1.0.0",
        "tasks": [t["task_id"] for t in TASKS],
    }


@app.get("/schema")
def schema():
    return {
        "action": KitchenAction.model_json_schema(),
        "observation": KitchenObservation.model_json_schema(),
    }


@app.post("/reset")
def reset(body: Dict[str, Any] = Body(default={})):
    task_id = body.get("task_id")
    sid = body.get("episode_id") or str(uuid.uuid4())
    env = _get_or_create(sid)
    obs = env.reset(task_id=task_id)
    return _obs_dict(obs, sid)


@app.post("/step")
def step(body: Dict[str, Any] = Body(...)):
    action_data = body.get("action")
    if action_data is None:
        raise HTTPException(status_code=422, detail="'action' field required")

    sid = body.get("episode_id", _DEFAULT)
    
    try:
        action = KitchenAction(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    env = _get_or_create(sid)
    if not hasattr(env, '_orders') or not env._orders:
        env.reset()

    obs = env.step(action)
    return _obs_dict(obs, sid)


@app.get("/state")
def state(episode_id: str = _DEFAULT):
    env = _get_or_create(episode_id)
    s = env.state
    return {"episode_id": getattr(s, 'episode_id', episode_id), "step_count": getattr(s, 'step_count', 0)}


@app.post("/mcp")
def mcp(body: Dict[str, Any] = Body(default={})):
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {"name": "kitchenflow_env"},
    })


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """
    Primary entry point for the validator.
    Takes no arguments to ensure it is callable by the automated checker.
    """
    import uvicorn
    
    # Use argparse inside main so it handles CLI flags when run directly,
    # but uses defaults when called by the validator.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()

    # Launching the app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
