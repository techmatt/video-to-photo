"""HTTP server backing the Export Labels button in faces_review.html.

Exposes a single endpoint `POST /export` that re-reads the per-corpus
`face_labels.json` (written by faces_review.html) and calls into
`save_labeled_faces.run_export()` to push new face crops into a global store.

Usage:
    uv run python -m still_extractor.launch_faces_export_server \\
        --config configs/june27.yaml \\
        [--output-dir data/ground_truth/face_labels] \\
        [--port 7432]
"""

import argparse
import json
import logging
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from still_extractor.inventory import RunConfig
from still_extractor.save_labeled_faces import _atomic_write_json, run_export

logger = logging.getLogger(__name__)


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _make_handler(
    labels_json_path: Path,
    results_path: Path,
    output_dir: Path,
    corpus_name: str,
):
    class ExportHandler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            if self.path != "/export":
                self.send_response(404)
                for k, v in CORS_HEADERS.items():
                    self.send_header(k, v)
                self.end_headers()
                return
            self.send_response(200)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/export":
                self._write_json(404, {"ok": False, "error": f"unknown path {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length > 0 else b""

                if body:
                    try:
                        posted_labels = json.loads(body.decode("utf-8"))
                    except Exception as e:
                        self._write_json(400, {
                            "ok": False,
                            "error": f"invalid JSON body: {e}",
                        })
                        return
                    if not isinstance(posted_labels, dict):
                        self._write_json(400, {
                            "ok": False,
                            "error": "body must be a JSON object mapping card_key -> label",
                        })
                        return
                    logger.info("Received %d labels from client; updating %s",
                                len(posted_labels), labels_json_path)
                    _atomic_write_json(labels_json_path, posted_labels)

                result = run_export(
                    labels_json_path=labels_json_path,
                    results_path=results_path,
                    output_dir=output_dir,
                    corpus_name=corpus_name,
                )
                payload = {"ok": True, **result}
                self._write_json(200, payload)
            except Exception as e:
                logger.exception("Export failed")
                err_body = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                self._write_json(500, err_body)

        def log_message(self, fmt: str, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    return ExportHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP server for exporting face labels from faces_review.html.",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Run YAML config. Determines labels-json path and corpus name.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ground_truth/face_labels"),
                        help="Global face labels store. Default: data/ground_truth/face_labels.")
    parser.add_argument("--port", type=int, default=7432,
                        help="TCP port to bind. Default: 7432.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    cfg = RunConfig.from_yaml(args.config)
    labels_json_path = (cfg.output_dir / "face_labels.json").resolve()
    results_path = (cfg.output_dir / "results.parquet").resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()

    if not results_path.exists():
        logger.error("results.parquet not found at %s", results_path)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler_cls = _make_handler(
        labels_json_path=labels_json_path,
        results_path=results_path,
        output_dir=output_dir,
        corpus_name=cfg.name,
    )

    server = HTTPServer(("localhost", args.port), handler_cls)

    print(f"Face export server running on http://localhost:{args.port}")
    print(f"Config:  {config_path}  (corpus: {cfg.name})")
    print(f"Labels:  {labels_json_path}")
    print(f"Store:   {output_dir}")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
