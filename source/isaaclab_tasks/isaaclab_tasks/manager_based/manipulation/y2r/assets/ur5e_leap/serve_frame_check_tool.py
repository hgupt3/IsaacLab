#!/usr/bin/env python3
"""Serve a browser viewer for checking UR5e LEAP palm/tip frames."""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml


ASSET_DIR = Path(__file__).resolve().parent
TOOL_DIR = ASSET_DIR / "frame_check_tool"
ROBOT_CONFIG_PATH = ASSET_DIR.parents[1] / "configs" / "robots" / "ur5e_leap.yaml"
HOST = "0.0.0.0"
PORT = 8766


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), FrameCheckHandler)
    print(f"Serving UR5e LEAP frame-check tool on {HOST}:{PORT}")
    server.serve_forever()


def load_frame_config() -> dict:
    with ROBOT_CONFIG_PATH.open("r") as f:
        config = yaml.safe_load(f)
    robot_cfg = config["robot"]
    return {
        "urdf": "ur5e_leap_right_gemini305.urdf",
        "palm": {
            "urdf_link": "palm_frame",
            "parent_link": robot_cfg["palm_body_name"],
            "offset": robot_cfg["palm_frame_offset"],
        },
        "tips": {
            "thumb": {
                "urdf_link": "thumb_tip",
                "parent_link": "thumb_link_3",
                "offset": robot_cfg["tip_offsets"]["thumb"],
            },
            "index": {
                "urdf_link": "index_tip",
                "parent_link": "index_link_3",
                "offset": robot_cfg["tip_offsets"]["index"],
            },
            "middle": {
                "urdf_link": "middle_tip",
                "parent_link": "middle_link_3",
                "offset": robot_cfg["tip_offsets"]["middle"],
            },
            "ring": {
                "urdf_link": "ring_tip",
                "parent_link": "ring_link_3",
                "offset": robot_cfg["tip_offsets"]["ring"],
            },
        },
        "contact_layout": robot_cfg["contact_layout"],
    }


class FrameCheckHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._serve_file(TOOL_DIR / "index.html", send_body=False)
        elif path.startswith("/asset/"):
            self._serve_asset(path, send_body=False)
        else:
            self.send_error(404, "not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                self._serve_file(TOOL_DIR / "index.html")
            elif path == "/api/frame_config":
                self._send_json(load_frame_config())
            elif path.startswith("/asset/"):
                self._serve_asset(path)
            else:
                self.send_error(404, "not found")
        except Exception as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=500)

    def _serve_asset(self, request_path: str, send_body: bool = True) -> None:
        rel_path = request_path.removeprefix("/asset/")
        target = (ASSET_DIR / rel_path).resolve()
        if not target.is_relative_to(ASSET_DIR):
            self.send_error(403, "forbidden")
            return
        self._serve_file(target, send_body=send_body)

    def _serve_file(self, path: Path, send_body: bool = True) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "not found")
            return
        content_type = mimetypes.guess_type(str(path))[0]
        if content_type is None:
            content_type = "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    main()
