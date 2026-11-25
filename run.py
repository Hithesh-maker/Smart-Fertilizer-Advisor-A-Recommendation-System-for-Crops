"""CLI entrypoint for launching Smart Fertilizer Advisor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from smart_fertilizer import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Smart Fertilizer Advisor v2 web server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5055")))
    args = parser.parse_args(argv)

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
