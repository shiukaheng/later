import json
import logging
import os
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from later.feed.synthetic import AuthorPool, engagement
from later.main import parse_args

PORT = int(os.environ.get("LATER_PORT", 8000))
WEB_BUILD = Path(__file__).with_name("web_dist")
log = logging.getLogger("later.web")
PAGES = {"/", "/dashboard", "/dashboard/"}


def ensure_frontend() -> Path:
    """Build the browser assets on first launch from a source checkout."""
    index = WEB_BUILD / "index.html"
    frontend = WEB_BUILD.resolve().parents[2] / "frontend"
    if index.is_file():
        sources = [frontend / name for name in ("bun.lock", "index.html", "vite.config.ts", "tsconfig.json")]
        sources.extend((frontend / "src").rglob("*"))
        if not any(source.is_file() and source.stat().st_mtime_ns > index.stat().st_mtime_ns for source in sources):
            return index
    if not (frontend / "bun.lock").is_file():
        raise SystemExit("Browser assets are missing. Run from a LATER source checkout or install a wheel with the frontend built.")
    bun = shutil.which("bun")
    if bun is None:
        raise SystemExit("Install Bun (https://bun.com/) to build the browser frontend on first launch.")
    log.info("building the browser frontend for first launch")
    try:
        subprocess.run([bun, "install", "--frozen-lockfile"], cwd=frontend, check=True)
        subprocess.run([bun, "run", "build"], cwd=frontend, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not build the browser frontend: {exc}") from exc
    if not index.is_file():
        raise SystemExit("Frontend build did not produce web_dist/index.html.")
    return index


def main() -> None:
    """Browser feed. Same backend (Session) as the terminal version."""
    args = parse_args("Browser feed", web=True)
    from later.session import Session

    logging.basicConfig(
        level=os.environ.get("LATER_LOG", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2"):  # OpenAI SDK per-request logs
        logging.getLogger(name).setLevel(logging.WARNING)
    index = ensure_frontend()
    session = Session(profile=args.profile)
    authors = AuthorPool()
    author_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, ctype: str = "application/json", code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if ctype.startswith("text/html"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def route(self) -> None:
            if self.command == "GET" and self.path.partition("?")[0] in PAGES:
                try:
                    page = index.read_bytes()
                except FileNotFoundError:
                    self.send(b"Frontend files are being rebuilt or missing. Reload after the build finishes.",
                              "text/plain; charset=utf-8", 503)
                else:
                    self.send(page, "text/html; charset=utf-8")
            elif self.command == "GET" and self.path.startswith("/assets/"):
                name = self.path.removeprefix("/assets/")
                asset = WEB_BUILD / "assets" / name
                types = {".js": "text/javascript", ".css": "text/css"}
                if name and "/" not in name and asset.suffix in types and asset.is_file():
                    self.send(asset.read_bytes(), types[asset.suffix])
                else:
                    self.send(b"not found", "text/plain", 404)
            elif self.command == "GET" and self.path == "/next":
                post = session.next()
                with author_lock:
                    a = authors.take()
                author = {"id": a["id"], "name": a["name"], "handle": a["handle"], "avatar": a["avatar"]}
                self.send(json.dumps({**post, "author": author, "stats": engagement()}).encode())
            elif self.command == "GET" and self.path == "/info":
                self.send(session.info().encode(), "text/plain; charset=utf-8")
            elif self.command == "GET" and self.path == "/api/policies":
                self.send(json.dumps(session.policy_snapshot()).encode())
            elif self.command == "GET" and self.path == "/api/policies/timeline":
                self.send(json.dumps(session.policy_timeline()).encode())
            elif self.command == "POST" and self.path == "/api/policies":
                self.change_policy()
            elif self.command == "POST" and self.path == "/rate":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                session.rate(body["id"], int(body["reward"]))
                self.send(b"{}")
            else:
                self.send(b"not found", "text/plain", 404)

        def change_policy(self) -> None:
            if self.headers.get_content_type() != "application/json":
                self.send(b'{"error":"Expected JSON."}', code=415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("Invalid request size.")
                body = json.loads(self.rfile.read(size))
                keys = ("action", "dimension", "option", "text", "field")
                if not isinstance(body, dict) or any(not isinstance(body.get(key, ""), str) for key in keys):
                    raise ValueError("Expected text fields.")
                session.change_option(body.get("action", ""), body.get("dimension", ""),
                                      option=body.get("option", ""), text=body.get("text", ""),
                                      field=body.get("field", ""), value=body.get("value"))
            except (ValueError, UnicodeError) as exc:
                self.send(json.dumps({"error": str(exc)}).encode(), code=400)
                return
            self.send(json.dumps(session.policy_snapshot()).encode())

        def dispatch(self) -> None:
            start = time.monotonic()
            try:
                self.route()
            except Exception:
                log.exception("%s %s failed", self.command, self.path)
                self.send(b'{"error": "server error, see terminal"}', code=500)
            if self.path != "/":
                log.debug("%s %s %.2fs", self.command, self.path, time.monotonic() - start)

        do_GET = do_POST = dispatch

        def log_message(self, *args):  # replaced by our own logging above
            pass

    server = ThreadingHTTPServer((args.host, PORT), Handler)
    log.info("LATER profile %s listening on %s:%d  (Ctrl-C to stop)", args.profile, args.host, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()
        print(f"\nsaved {session.rating_count} ratings to {session.path}")
