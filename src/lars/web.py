import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lars import store
from lars.session import Session

PORT = int(os.environ.get("LARS_PORT", 8000))
PAGE = (Path(__file__).parent / "web.html").read_bytes()


def main() -> None:
    """Browser feed. Same backend (Session) as the terminal version."""
    session = Session()

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/next":
                tid, tweet = session.next()
                self.send(json.dumps({"id": tid, "text": tweet}).encode())
            elif self.path == "/info":
                self.send(session.info().encode(), "text/plain; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/rate":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                session.rate(body["id"], int(body["reward"]))
                self.send(b"{}")
            else:
                self.send_error(404)

        def log_message(self, *args):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"LARS feed on http://localhost:{PORT}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()
        print(f"\nsaved {len(session.log)} ratings to {store.PATH}")
