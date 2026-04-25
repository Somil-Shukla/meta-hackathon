"""
FastAPI application for the Fraud Detection Environment.

Creates an HTTP server exposing the FraudEnvironment over HTTP + WebSocket
endpoints compatible with EnvClient.

Endpoints:
    POST /reset   — Reset the environment (body may include task_name, seed)
    POST /step    — Execute a combined defender+fraudster action
    GET  /state   — Get current environment state (full hidden state)
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    uv run server                          # via pyproject.toml entry point
    uvicorn server.app:app --reload        # dev mode
    python -m scam_detection.server.app    # direct
"""
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with:\n    uv sync"
    ) from e

try:
    from scam_detection.models import FraudAction, FraudObservation
    from scam_detection.server.fraud_environment import FraudEnvironment
except (ModuleNotFoundError, ImportError):
    try:
        from ..models import FraudAction, FraudObservation
        from .fraud_environment import FraudEnvironment
    except ImportError:
        from models import FraudAction, FraudObservation
        from fraud_environment import FraudEnvironment

app = create_app(
    FraudEnvironment,
    FraudAction,
    FraudObservation,
    env_name="fraud_detection",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Entry point for ``uv run server`` or ``python -m ...``."""
    if port is None:
        port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Environment Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
