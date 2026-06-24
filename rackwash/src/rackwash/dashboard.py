"""Local web dashboard: live MJPEG feed + You-vs-Wife scoreboard over a
WebSocket. Reads SharedState snapshots; never touches the camera directly."""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from rackwash.state import SharedState

_INDEX_HTML = """<!doctype html>
<html><head><title>Dish Counter</title>
<style>
 body{font-family:system-ui;background:#111;color:#eee;text-align:center}
 .board{display:flex;justify-content:center;gap:3rem;margin:1rem}
 .name{font-size:1.2rem;opacity:.7} .num{font-size:3rem;font-weight:700}
 #view{overflow:hidden;display:inline-block;max-width:90vw;
       border:2px solid #333;border-radius:8px;line-height:0}
 #feed{max-width:90vw;transform-origin:center;transition:transform .1s}
 .zoom{margin:.5rem}
 .zoom button{font-size:1rem;padding:.3rem .8rem;margin:0 .2rem;cursor:pointer}
 #status{color:#f55}
</style></head>
<body>
 <h1>Dish Counter</h1>
 <p id="status"></p>
 <div id="view"><img src="/stream" alt="live feed" id="feed"/></div>
 <div class="zoom">
  <button onclick="zoom(-0.5)">Zoom &minus;</button>
  <span id="zlevel">1.0&times;</span>
  <button onclick="zoom(0.5)">Zoom +</button>
  <button onclick="zoomReset()">Reset</button>
 </div>
 <div class="zoom">
  <button onclick="toggleDetections()" id="detbtn">Show detections</button>
  <button onclick="resetCounts()">Reset counts</button>
 </div>
 <p id="detected"></p>
 <div class="board">
  <div><div class="name">You</div><div class="num" id="you">0</div>
       <div id="you-all">all-time 0</div></div>
  <div><div class="name">Wife</div><div class="num" id="wife">0</div>
       <div id="wife-all">all-time 0</div></div>
 </div>
<script>
 let z = 1;
 const feed = document.getElementById('feed');
 const zl = document.getElementById('zlevel');
 function applyZoom(){ feed.style.transform = 'scale(' + z + ')';
                       zl.textContent = z.toFixed(1) + '×'; }
 function zoom(d){ z = Math.min(4, Math.max(1, Math.round((z + d) * 10) / 10));
                   applyZoom(); }
 function zoomReset(){ z = 1; applyZoom(); }
 applyZoom();

 function toggleDetections(){ fetch('/detections', {method:'POST'}); }
 function resetCounts(){
   if (confirm('Reset all counts (today and all-time)?'))
     fetch('/reset', {method:'POST'});
 }

 const ws = new WebSocket(`ws://${location.host}/ws`);
 ws.onmessage = (e) => {
   const d = JSON.parse(e.data);
   const t = (d.counts && d.counts.today) || {You:0,Wife:0};
   const a = (d.counts && d.counts.all_time) || {You:0,Wife:0};
   you.textContent = t.You; wife.textContent = t.Wife;
   document.getElementById('you-all').textContent = 'all-time ' + a.You;
   document.getElementById('wife-all').textContent = 'all-time ' + a.Wife;
   status.textContent = d.camera_online ? '' : 'camera offline';
   document.getElementById('detbtn').textContent =
     d.show_detections ? 'Hide detections' : 'Show detections';
   document.getElementById('detected').textContent =
     d.show_detections ? ('detected: ' + ((d.detections || []).join(', ') || 'none')) : '';
 };
</script>
</body></html>
"""


def create_app(state: SharedState, store=None) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/counts")
    def counts() -> JSONResponse:
        return JSONResponse(state.snapshot()["counts"])

    @app.post("/reset")
    def reset() -> JSONResponse:
        if store is not None:
            store.reset(time.time())
        return JSONResponse({"ok": True})

    @app.post("/detections")
    def detections() -> JSONResponse:
        return JSONResponse({"show_detections": state.toggle_detections()})

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def frames():
            while True:
                jpeg = state.latest_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                time.sleep(0.05)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                snap = state.snapshot()
                await socket.send_json(
                    {
                        "counts": snap["counts"],
                        "camera_online": snap["camera_online"],
                        "show_detections": snap["show_detections"],
                        "detections": snap["detections"],
                    }
                )
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def run_server(
    state: SharedState, store=None, host: str = "127.0.0.1", port: int = 8000
) -> None:
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(state, store), host=host, port=port, log_level="warning")
