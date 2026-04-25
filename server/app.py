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
    """
    Entry point for ``uv run server``, ``python -m ...``, or direct call.

    Argument resolution order (highest priority first):
      1. CLI flags   --host / --port
      2. Function arguments (when called programmatically)
      3. PORT environment variable
      4. Hard-coded default: 8000
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fraud Detection Environment Server",
        # Allow unknown args so uvicorn sub-options don't crash startup
        add_help=True,
    )
    parser.add_argument("--host", type=str, default=None,
                        help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Bind port (default: PORT env or 8000)")

    # parse_known_args so leftover uv/pip flags are ignored gracefully
    args, _ = parser.parse_known_args(sys.argv[1:])

    resolved_host = args.host or host
    resolved_port = args.port or port or int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    main()
