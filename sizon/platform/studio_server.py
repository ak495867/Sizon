"""Local-only Studio server; bind explicitly when exposing beyond localhost."""
from __future__ import annotations
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from sizon.platform.production import HealthMonitor
class StudioHandler(SimpleHTTPRequestHandler):
    monitor=HealthMonitor()
    def do_GET(self):
        if self.path=="/health": self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(str(self.monitor.health()).replace("'","\"").encode()); return
        return super().do_GET()
def serve(directory="runs",host="127.0.0.1",port=8765):
    directory=str(Path(directory).resolve()); import os; os.chdir(directory); return ThreadingHTTPServer((host,port),StudioHandler)
