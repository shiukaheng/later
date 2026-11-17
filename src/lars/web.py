import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lars.arguments import parse_args
from lars.session import Session

PORT = int(os.environ.get("LARS_PORT", 8000))
PAGE = (Path(__file__).parent / "web.html").read_bytes()
log = logging.getLogger("lars.web")


def main() -> None:
    """Browser feed. Same backend (Session) as the terminal version."""
    args = parse_args("Browser feed", web=True)
    logging.basicConfig(
        level=os.environ.get("LARS_LOG", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2"):  # OpenAI SDK per-request logs
        logging.getLogger(name).setLevel(logging.WARNING)
    session = Session(profile=args.profile)

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, ctype: str = "application/json", code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def route(self) -> None:
            if self.command == "GET" and self.path == "/":
                self.send(PAGE, "text/html; charset=utf-8")
            elif self.command == "GET" and self.path == "/next":
                post = session.next()
                a = post["author"]
                author = {"id": a["id"], "name": a["name"], "handle": a["handle"], "avatar": a["avatar"]}
                self.send(json.dumps({**post, "author": author}).encode())
            elif self.command == "GET" and self.path == "/info":
                self.send(session.info().encode(), "text/plain; charset=utf-8")
            elif self.command == "POST" and self.path == "/rate":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                session.rate(body["id"], int(body["reward"]))
                self.send(b"{}")
            else:
                self.send(b"not found", "text/plain", 404)

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
    log.info("LARS profile %s listening on %s:%d  (Ctrl-C to stop)", args.profile, args.host, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()
        print(f"\nsaved {len(session.log)} ratings to {session.path}")
