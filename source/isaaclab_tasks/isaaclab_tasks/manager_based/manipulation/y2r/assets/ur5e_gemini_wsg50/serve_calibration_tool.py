#!/usr/bin/env python3
"""Serve the UR5e Gemini WSG50 calibration editor."""

from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml


ASSET_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = ASSET_DIR / "calibration.yaml"
BUILD_SCRIPT = ASSET_DIR / "build_ur5e_gemini_wsg50.py"
TOOL_DIR = ASSET_DIR / "calibration_tool"


def main() -> None:
    config = load_calibration()
    host = str(config["editor"]["host"])
    port = int(config["editor"]["port"])
    server = ThreadingHTTPServer((host, port), CalibrationHandler)
    print(f"Serving UR5e Gemini WSG50 calibration tool on {host}:{port}")
    server.serve_forever()


def load_calibration() -> dict:
    with CALIBRATION_PATH.open("r") as f:
        return yaml.safe_load(f)


def save_calibration(config: dict) -> None:
    with CALIBRATION_PATH.open("w") as f:
        yaml.dump(config, f, Dumper=FlowListDumper, sort_keys=False)


class FlowListDumper(yaml.SafeDumper):
    pass


def represent_list(dumper: yaml.SafeDumper, data: list):
    if len(data) == 3 and all(isinstance(value, (int, float)) for value in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


FlowListDumper.add_representer(list, represent_list)


class CalibrationHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                self._serve_file(TOOL_DIR / "index.html", send_body=False)
            elif path.startswith("/asset/"):
                self._serve_asset(path, send_body=False)
            else:
                self.send_error(404, "not found")
        except Exception as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=500)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                self._serve_file(TOOL_DIR / "index.html")
            elif path == "/api/calibration":
                self._send_json(load_calibration())
            elif path.startswith("/asset/"):
                self._serve_asset(path)
            else:
                self.send_error(404, "not found")
        except Exception as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/calibration":
            self.send_error(404, "not found")
            return

        try:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            config = load_calibration()
            config["transforms"] = payload["transforms"]
            config["default_pose"] = payload["default_pose"]
            config["contact_markers"] = payload["contact_markers"]
            save_calibration(config)
            proc = subprocess.run(
                [sys.executable, str(BUILD_SCRIPT)],
                cwd=str(ASSET_DIR),
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                self._send_json(
                    {
                        "status": "error",
                        "return_code": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    },
                    status=400,
                )
                return
            self._send_json({"status": "ok", "stdout": proc.stdout, "stderr": proc.stderr})
        except Exception as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=400)

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
