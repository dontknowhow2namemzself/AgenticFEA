"""Pictures reach the browser: an image through Streamlit's gzip middleware,
requested the way a browser asks for it.

The page's pictures (the mesh, the result views) are served from /media/
through this middleware. With Starlette 1.8.0 it sent the body before the
response start, so every picture failed to load while the PDF report, which
reads the files directly, still had them. The tests in test_dashboard.py do not
go through the HTTP server, so they could not see it.
"""

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient
from streamlit.web.server.starlette.starlette_gzip_middleware import SelectiveGZipMiddleware

PNG = b"\x89PNG\r\n\x1a\n" + bytes(4000)          # above the 500-byte compression minimum


def test_an_image_reaches_the_browser_intact():
    app = Starlette(routes=[Route("/media/x.png", lambda request: Response(PNG, media_type="image/png"))])
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=500)
    r = TestClient(app).get("/media/x.png", headers={"Accept-Encoding": "gzip, deflate, br, zstd"})
    assert r.status_code == 200
    assert r.content == PNG
