from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {"ui"}
ALLOWED_SUBTREES = {("outputs", "metrics"), ("outputs", "figures")}
ALLOWED_FILES = {("report", "submission_report.md")}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path == "/":
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        relative = PurePosixPath(unquote(urlsplit(path).path).lstrip("/"))
        parts = relative.parts
        if ".." in parts or not parts:
            return str(ROOT / "__not_found__")
        is_allowed = (
            parts[0] in ALLOWED_TOP
            or tuple(parts[:2]) in ALLOWED_SUBTREES
            or tuple(parts) in ALLOWED_FILES
        )
        if not is_allowed:
            return str(ROOT / "__not_found__")
        return str(ROOT.joinpath(*parts))

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local results dashboard without exposing source data.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Dashboard: http://127.0.0.1:{args.port}/ui/")
    print("Only ui/, outputs/metrics/, outputs/figures/, and the submission report are served.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
