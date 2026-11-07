"""CLI entrypoint to launch the MarketPulse FastAPI application server."""

import sys
import argparse
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MarketPulse FastAPI Backend Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code modifications")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Starting MarketPulse Demand Forecasting API Server on http://{args.host}:{args.port}")
    print(f"Interactive API Documentation: http://{args.host}:{args.port}/docs")
    print("=" * 80)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

