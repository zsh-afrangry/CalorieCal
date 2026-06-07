from __future__ import annotations

import argparse
import ctypes
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from openni_depth_probe import (
    ONI_API_VERSION,
    ONI_SENSOR_DEPTH,
    OpenNI,
    OniDeviceHandle,
    OniFramePtr,
    OniStreamHandle,
    get_nonzero_median,
    require_ok,
    try_enable_registration,
)


class DepthReader:
    def __init__(
        self,
        openni_root: Path,
        *,
        enable_registration: bool = False,
    ) -> None:
        self.openni_root = openni_root
        self.enable_registration = enable_registration
        self.openni: OpenNI | None = None
        self.device = OniDeviceHandle()
        self.stream = OniStreamHandle()
        self.stream_started = False
        self.lock = threading.Lock()

    def start(self) -> None:
        self.openni = OpenNI(self.openni_root)
        print(f"openni_dll={self.openni.dll_path}")
        require_ok(self.openni.dll.oniInitialize(ONI_API_VERSION), "oniInitialize")
        require_ok(
            self.openni.dll.oniDeviceOpen(None, ctypes.byref(self.device)),
            "oniDeviceOpen",
        )

        if self.enable_registration:
            try_enable_registration(self.openni, self.device)

        require_ok(
            self.openni.dll.oniDeviceCreateStream(
                self.device,
                ONI_SENSOR_DEPTH,
                ctypes.byref(self.stream),
            ),
            "oniDeviceCreateStream(depth)",
        )
        require_ok(self.openni.dll.oniStreamStart(self.stream), "oniStreamStart(depth)")
        self.stream_started = True

    def close(self) -> None:
        if self.openni is None:
            return

        if self.stream_started:
            self.openni.dll.oniStreamStop(self.stream)
            self.stream_started = False

        if self.stream:
            self.openni.dll.oniStreamDestroy(self.stream)

        if self.device:
            self.openni.dll.oniDeviceClose(self.device)

        self.openni.dll.oniShutdown()
        self.openni.close_dll_directory()
        self.openni = None

    def read_depth(self) -> np.ndarray:
        if self.openni is None:
            raise RuntimeError("DepthReader is not started.")

        with self.lock:
            frame_ptr = OniFramePtr()
            require_ok(
                self.openni.dll.oniStreamReadFrame(self.stream, ctypes.byref(frame_ptr)),
                "oniStreamReadFrame(depth)",
            )

            try:
                frame = frame_ptr.contents
                byte_count = frame.dataSize
                raw = (ctypes.c_uint16 * (byte_count // 2)).from_address(frame.data)
                depth = np.ctypeslib.as_array(raw).reshape(frame.height, frame.width)
                return depth.copy()
            finally:
                self.openni.dll.oniFrameRelease(frame_ptr)


def clamp_pixel(value: int, max_value: int) -> int:
    return min(max(value, 0), max_value)


def resolve_point(payload: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    norm_x = payload.get("normX", payload.get("norm_x"))
    norm_y = payload.get("normY", payload.get("norm_y"))

    if norm_x is not None and norm_y is not None:
        x = round(float(norm_x) * (width - 1))
        y = round(float(norm_y) * (height - 1))
    else:
        x = int(payload.get("x", width // 2))
        y = int(payload.get("y", height // 2))

    return clamp_pixel(x, width - 1), clamp_pixel(y, height - 1)


def read_payload_depth(
    depth: np.ndarray,
    payload: dict[str, Any],
    default_radius: int,
    default_search_radius: int,
) -> dict[str, Any]:
    height, width = depth.shape
    x, y = resolve_point(payload, width, height)
    radius = int(payload.get("radius", default_radius))
    search_radius = int(payload.get("searchRadius", default_search_radius))
    raw_depth = int(depth[y, x])
    median_depth = get_nonzero_median(depth, x, y, radius)
    foreground_depth = get_percentile_depth(depth, x, y, search_radius, 15)
    valid_ratio = get_valid_ratio(depth, x, y, search_radius)

    return {
        "id": payload.get("id"),
        "x": x,
        "y": y,
        "radius": radius,
        "searchRadius": search_radius,
        "depthMm": raw_depth,
        "medianDepthMm": median_depth,
        "foregroundDepthMm": foreground_depth,
        "validRatio": valid_ratio,
        "foregroundCandidate": (
            foreground_depth > 0 and median_depth > 0 and foreground_depth < median_depth
        ),
        "valid": median_depth > 0 or foreground_depth > 0,
    }


def get_patch(depth: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    height, width = depth.shape
    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    return depth[y1:y2, x1:x2]


def get_percentile_depth(
    depth: np.ndarray,
    x: int,
    y: int,
    radius: int,
    percentile: int,
) -> int:
    patch = get_patch(depth, x, y, radius)
    values = patch[patch > 0]

    if values.size == 0:
        return 0

    return int(np.percentile(values, percentile))


def get_valid_ratio(depth: np.ndarray, x: int, y: int, radius: int) -> float:
    patch = get_patch(depth, x, y, radius)

    if patch.size == 0:
        return 0.0

    return round(float(np.count_nonzero(patch > 0) / patch.size), 3)


class DepthRequestHandler(BaseHTTPRequestHandler):
    reader: DepthReader
    default_radius: int
    default_search_radius: int

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.write_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self.respond_json(
                {
                    "ok": True,
                    "service": "openni-depth-server",
                    "time": time.time(),
                },
            )
            return

        if parsed.path != "/depth":
            self.respond_json({"ok": False, "error": "not found"}, status=404)
            return

        query = parse_qs(parsed.query)
        payload = {
            "x": first_value(query, "x"),
            "y": first_value(query, "y"),
            "normX": first_value(query, "normX", "norm-x", "norm_x"),
            "normY": first_value(query, "normY", "norm-y", "norm_y"),
            "radius": first_value(query, "radius"),
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        self.respond_depth([payload])

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path != "/depth":
            self.respond_json({"ok": False, "error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as error:
            self.respond_json({"ok": False, "error": str(error)}, status=400)
            return

        points = payload.get("points")

        if points is None:
            points = [payload]

        if not isinstance(points, list):
            self.respond_json({"ok": False, "error": "points must be a list"}, status=400)
            return

        self.respond_depth(points)

    def respond_depth(self, points: list[dict[str, Any]]) -> None:
        try:
            depth = self.reader.read_depth()
            height, width = depth.shape
            results = [
                read_payload_depth(
                    depth,
                    point,
                    self.default_radius,
                    self.default_search_radius,
                )
                for point in points
                if isinstance(point, dict)
            ]
            self.respond_json(
                {
                    "ok": True,
                    "width": width,
                    "height": height,
                    "points": results,
                    "time": time.time(),
                },
            )
        except Exception as error:
            self.respond_json({"ok": False, "error": str(error)}, status=500)

    def respond_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def first_value(query: dict[str, list[str]], *keys: str) -> str | None:
    for key in keys:
        values = query.get(key)

        if values:
            return values[0]

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenNI depth lookup API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--openni-redist", default="OpenNI_2.3.0.86")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--search-radius", type=int, default=35)
    parser.add_argument("--enable-registration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = DepthReader(
        Path(args.openni_redist).resolve(),
        enable_registration=args.enable_registration,
    )
    reader.start()

    DepthRequestHandler.reader = reader
    DepthRequestHandler.default_radius = args.radius
    DepthRequestHandler.default_search_radius = args.search_radius
    server = ThreadingHTTPServer((args.host, args.port), DepthRequestHandler)

    print(f"depth_server=http://{args.host}:{args.port}")
    print("GET  /health")
    print("GET  /depth?normX=0.5&normY=0.5")
    print("POST /depth {\"points\":[{\"id\":\"leftWrist\",\"normX\":0.5,\"normY\":0.5}]}")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        reader.close()


if __name__ == "__main__":
    main()
