"""
Main entrypoint for the AODLF backend.
Supports launching the interactive CLI or running the FastAPI SSE server.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to enable imports
current_dir = Path(__file__).resolve().parent
gen_root = current_dir.parent.parent
if str(gen_root) not in sys.path:
    sys.path.insert(0, str(gen_root))

from app.cli.interface import start_cli


def main():
    parser = argparse.ArgumentParser(description="AODLF Backend Entrypoint")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run FastAPI REST & SSE server instead of CLI mode"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    if args.server:
        import uvicorn
        from app.api.routes import app
        print(f"Starting AODLF FastAPI Backend Server on http://{args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Launch Interactive CLI App
        start_cli()


if __name__ == "__main__":
    main()
