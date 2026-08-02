#!/usr/bin/env python3
"""Automated local integrity test for Human Attributes Detector.

This script intentionally excludes Docker and MiniCPM inference. It exercises
the CPU installation, unit tests, PyTorch, the trained Transformer head, the
real YOLO and ResNet weights, and the complete HTTP video pipeline.
"""

from __future__ import annotations

import argparse
import compileall
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ATTRIBUTE = "не определен"


class TestFailure(RuntimeError):
    """A failure with a concise, user-facing explanation."""


@dataclass(slots=True)
class StepResult:
    name: str
    status: str
    seconds: float
    detail: str = ""


class TestRun:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results: list[StepResult] = []

    def step(self, name: str, action: Callable[[], str | None]) -> None:
        number = len(self.results) + 1
        print(f"\n[{number}] {name}", flush=True)
        started = time.monotonic()
        try:
            detail = action() or ""
        except Exception as exc:
            elapsed = time.monotonic() - started
            detail = str(exc) or exc.__class__.__name__
            self.results.append(StepResult(name, "FAIL", elapsed, detail))
            print(f"    FAIL ({elapsed:.1f}s): {detail}", flush=True)
            raise
        elapsed = time.monotonic() - started
        self.results.append(StepResult(name, "PASS", elapsed, detail))
        suffix = f": {detail}" if detail else ""
        print(f"    PASS ({elapsed:.1f}s){suffix}", flush=True)

    def save_report(self, error: str | None = None) -> Path:
        report_path = self.output_dir / "report.json"
        payload = {
            "success": error is None and all(item.status == "PASS" for item in self.results),
            "python": sys.version,
            "executable": sys.executable,
            "project_root": str(PROJECT_ROOT),
            "created_at": datetime.now().astimezone().isoformat(),
            "steps": [asdict(item) for item in self.results],
            "error": error,
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path


def run_command(command: list[str], *, timeout: int = 900) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.strip()
    if output:
        print("\n".join(f"      {line}" for line in output.splitlines()), flush=True)
    if completed.returncode:
        command_text = " ".join(command)
        raise TestFailure(f"command exited with {completed.returncode}: {command_text}")
    return output


def resolve_file(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise TestFailure(f"{description} not found: {path}")
    return path


def find_default_model(candidates: list[str], description: str) -> Path:
    for candidate in candidates:
        path = PROJECT_ROOT / "models" / candidate
        if path.is_file():
            return path.resolve()
    names = ", ".join(candidates)
    raise TestFailure(f"{description} not found in models/ (looked for: {names})")


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    expected_status: int = 200,
    timeout: float = 10,
) -> dict[str, object]:
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = exc.read()
    if status != expected_status:
        text = payload.decode("utf-8", errors="replace")[:1000]
        raise TestFailure(f"HTTP {status}, expected {expected_status}: {text}")
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TestFailure("server returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise TestFailure("server returned JSON that is not an object")
    return value


def multipart_bytes(field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----had-test-{secrets.token_hex(12)}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    return prefix + content + suffix, boundary


def upload_video(port: int, api_key: str, video: Path) -> dict[str, object]:
    boundary = f"----had-test-{secrets.token_hex(12)}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{video.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    content_length = len(prefix) + video.stat().st_size + len(suffix)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.putrequest("POST", "/api/v1/jobs")
        connection.putheader("X-API-Key", api_key)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(prefix)
        with video.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        payload = response.read()
        if response.status != 202:
            text = payload.decode("utf-8", errors="replace")[:1000]
            raise TestFailure(f"upload returned HTTP {response.status}: {text}")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TestFailure("upload response is not a JSON object")
        return value
    finally:
        connection.close()


def download_result(port: int, api_key: str, job_id: str, destination: Path) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(
            "GET", f"/api/v1/jobs/{job_id}/result", headers={"X-API-Key": api_key}
        )
        response = connection.getresponse()
        if response.status != 200:
            text = response.read().decode("utf-8", errors="replace")[:1000]
            raise TestFailure(f"result download returned HTTP {response.status}: {text}")
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    finally:
        connection.close()


def stop_process(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_server(base_url: str, process: subprocess.Popen[object], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise TestFailure(f"API process stopped with exit code {process.returncode}")
        try:
            health = request_json(f"{base_url}/health", timeout=2)
            if health.get("status") == "ok":
                return
            last_error = f"unexpected health response: {health}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TestFailure(f"API did not become ready: {last_error}")


def validate_metadata(path: Path, *, require_attributes: bool) -> str:
    if not path.is_file():
        raise TestFailure(f"metadata file was not created: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TestFailure(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(record, dict) or not {"frame", "time_seconds", "people"} <= record.keys():
            raise TestFailure(f"invalid frame record at line {line_number}")
        records.append(record)
    if not records:
        raise TestFailure("metadata JSONL is empty")

    detections = 0
    track_ids: set[int] = set()
    records_with_known_attributes = 0
    for record in records:
        people = record["people"]
        if not isinstance(people, list):
            raise TestFailure("metadata field 'people' is not a list")
        for person in people:
            if not isinstance(person, dict):
                raise TestFailure("person entry is not an object")
            required = {"track_id", "box", "confidence", "attributes"}
            if not required <= person.keys():
                raise TestFailure("person entry is missing required fields")
            if not isinstance(person["box"], list) or len(person["box"]) != 4:
                raise TestFailure("person box must contain four coordinates")
            if not isinstance(person["attributes"], dict):
                raise TestFailure("person attributes must be an object")
            detections += 1
            track_ids.add(int(person["track_id"]))
            if any(value != UNKNOWN_ATTRIBUTE for value in person["attributes"].values()):
                records_with_known_attributes += 1

    if detections == 0:
        raise TestFailure(
            "YOLO found no people; use a 5-15 second video with one clearly visible person"
        )
    if require_attributes and records_with_known_attributes == 0:
        raise TestFailure(
            "ResNet loaded, but no predicted attributes reached JSONL; use a longer video or "
            "inspect asynchronous attribute inference"
        )
    return (
        f"frames={len(records)}, detections={detections}, "
        f"track_ids={sorted(track_ids)}, attributed={records_with_known_attributes}"
    )


def validate_video(path: Path) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise TestFailure("OpenCV is unavailable; install requirements/cpu.txt") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise TestFailure(f"result video cannot be opened: {path}")
        frames = 0
        while frames < 3:
            ok, _frame = capture.read()
            if not ok:
                break
            frames += 1
        if frames == 0:
            raise TestFailure("result video contains no readable frames")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if path.stat().st_size == 0:
        raise TestFailure("result video is empty")
    return f"{width}x{height}, fps={fps:.2f}, size={path.stat().st_size} bytes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local unit, PyTorch, real-model, API, MP4, and JSONL checks. "
            "Docker and MiniCPM are intentionally excluded."
        )
    )
    parser.add_argument("--video", required=True, help="Path to a short video with a person")
    parser.add_argument("--yolo", help="YOLO ONNX path; known names in models/ are auto-detected")
    parser.add_argument(
        "--resnet", help="ResNet checkpoint path; known names in models/ are auto-detected"
    )
    parser.add_argument(
        "--transformer",
        help=(
            "VisionAttrTransformer checkpoint path; known legacy names in models/ "
            "are auto-detected"
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N (default: cpu)")
    parser.add_argument(
        "--backend", choices=("resnet", "none"), default="resnet", help="API backend"
    )
    parser.add_argument("--timeout", type=int, default=900, help="Job timeout in seconds")
    parser.add_argument(
        "--skip-unit", action="store_true", help="Skip compileall, pip check, and unit tests"
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Do not test the trained MiniCPM-embedding Transformer checkpoint",
    )
    parser.add_argument(
        "--with-ruff", action="store_true", help="Also run Ruff (requires development extras)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = PROJECT_ROOT / "var" / "system-tests" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    run = TestRun(output_dir)
    error: str | None = None

    try:
        video = resolve_file(args.video, "test video")
        yolo = (
            resolve_file(args.yolo, "YOLO model")
            if args.yolo
            else find_default_model(
                ["yolov8s_576x1024_v2.onnx", "yolo.onnx"], "YOLO model"
            )
        )
        resnet = None
        if args.backend == "resnet":
            resnet = (
                resolve_file(args.resnet, "ResNet checkpoint")
                if args.resnet
                else find_default_model(
                    ["resnet_ens_11.19_e60_s0.782.pt", "resnet_attributes.pt"],
                    "ResNet checkpoint",
                )
            )
        transformer = None
        if not args.skip_transformer:
            if args.transformer:
                transformer = resolve_file(args.transformer, "Transformer checkpoint")
            else:
                for candidate in (
                    "MiniCPM-2.6int4 weights.pt",
                    "MiniCPM-o 2.6int4 weights.pt",
                    "best_checkpoint.pt",
                    "vision_attr_transformer.pt",
                ):
                    path = PROJECT_ROOT / "models" / candidate
                    if path.is_file():
                        transformer = path.resolve()
                        break

        def preflight() -> str:
            if not (3, 11) <= sys.version_info[:2] < (3, 13):
                raise TestFailure("Python 3.11 or 3.12 is required")
            imports = ["numpy", "PIL", "fastapi", "uvicorn", "cv2", "onnxruntime", "torch"]
            code = (
                "import importlib; "
                f"mods={imports!r}; "
                "[importlib.import_module(name) for name in mods]; "
                "import silhouette_detector, SFSORT; "
                "print('imports:', ', '.join(mods + ['silhouette_detector', 'SFSORT']))"
            )
            run_command([sys.executable, "-c", code], timeout=120)
            return f"python={sys.version.split()[0]}, executable={sys.executable}"

        run.step("Environment and package imports", preflight)

        if not args.skip_unit:
            run.step(
                "Compile Python sources",
                lambda: "compiled"
                if compileall.compile_dir(PROJECT_ROOT / "src", quiet=1)
                and compileall.compile_dir(PROJECT_ROOT / "tests", quiet=1)
                else (_ for _ in ()).throw(TestFailure("compileall reported errors")),
            )
            run.step(
                "Installed dependency consistency",
                lambda: run_command([sys.executable, "-m", "pip", "check"], timeout=120)
                or "no broken requirements",
            )
            run.step(
                "Unit tests including PyTorch tests",
                lambda: run_command(
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-v",
                    ],
                    timeout=args.timeout,
                ).splitlines()[-1],
            )
        if args.with_ruff:
            run.step(
                "Ruff",
                lambda: run_command([sys.executable, "-m", "ruff", "check", "."], timeout=120)
                or "all checks passed",
            )

        def torch_runtime() -> str:
            code = (
                "import torch; "
                "print('torch:', torch.__version__); "
                "print('cuda_available:', torch.cuda.is_available()); "
                "print('cuda_version:', torch.version.cuda); "
                f"assert {args.device!r} == 'cpu' or torch.cuda.is_available(), "
                "'CUDA device requested but PyTorch CUDA is unavailable'"
            )
            output = run_command([sys.executable, "-c", code], timeout=120)
            return output.replace("\n", "; ")

        run.step("PyTorch runtime", torch_runtime)

        if transformer is not None:

            def transformer_checkpoint() -> str:
                code = """
import json
import sys
from pathlib import Path
import torch
from silhouette_detector.attributes.labels import ATTRIBUTE_SIZES_RU
from silhouette_detector.attributes.minicpm import _load_transformer_checkpoint
from silhouette_detector.attributes.transformer import VisionAttrTransformer

state, config = _load_transformer_checkpoint(Path(sys.argv[1]))
input_dim = int(state["input_proj.weight"].shape[1])
model = VisionAttrTransformer(
    input_dim=input_dim,
    hidden_dim=int(config.get("hidden_dim", 768)),
    num_heads=int(config.get("num_heads", 12)),
    num_layers=int(config.get("num_layers", 6)),
    attr_sizes=config.get("attr_sizes", ATTRIBUTE_SIZES_RU),
    drop_path_rate=float(config.get("drop_path_rate", 0.1)),
)
model.load_state_dict(state, strict=True)
model.to(sys.argv[2]).eval()
embeddings = torch.randn(1, 1, 1, input_dim, device=sys.argv[2])
mask = torch.ones((1, 1), dtype=torch.bool, device=sys.argv[2])
with torch.inference_mode():
    output = model(embeddings, mask)
shapes = {name: list(value.shape) for name, value in output.items()}
if not shapes or any(shape[0] != 1 for shape in shapes.values()):
    raise RuntimeError(f"unexpected Transformer output: {shapes}")
print(json.dumps({"input_dim": input_dim, "outputs": shapes}, ensure_ascii=False))
"""
                output = run_command(
                    [sys.executable, "-c", code, str(transformer), args.device],
                    timeout=args.timeout,
                )
                return output.splitlines()[-1]

            run.step(
                "Trained MiniCPM-embedding Transformer checkpoint",
                transformer_checkpoint,
            )
        elif not args.skip_transformer:
            print(
                "\n[info] Transformer checkpoint was not found; pass --transformer to test it.",
                flush=True,
            )

        def yolo_load() -> str:
            code = """
import json
import sys
from pathlib import Path
from silhouette_detector.detection import YoloOnnxDetector
model = YoloOnnxDetector(Path(sys.argv[1]), sys.argv[2])
print(json.dumps({"providers": model.providers, "input": [model.input_width, model.input_height]}))
"""
            output = run_command([sys.executable, "-c", code, str(yolo), args.device], timeout=180)
            return output.splitlines()[-1]

        run.step("Load real YOLO ONNX model", yolo_load)

        if resnet is not None:

            def resnet_predict() -> str:
                code = """
import json
import sys
from pathlib import Path
import cv2
from PIL import Image
from silhouette_detector.attributes.labels import ATTRIBUTE_KEYS
from silhouette_detector.attributes.resnet import ResNetBackend

capture = cv2.VideoCapture(sys.argv[2])
ok, frame = capture.read()
capture.release()
if not ok:
    raise RuntimeError("cannot read the first frame from the test video")
backend = ResNetBackend(Path(sys.argv[1]), sys.argv[3])
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
prediction = backend.predict(Image.fromarray(rgb))
if set(prediction) != set(ATTRIBUTE_KEYS):
    raise RuntimeError(f"unexpected attribute keys: {sorted(prediction)}")
print(json.dumps(prediction, ensure_ascii=False))
"""
                output = run_command(
                    [sys.executable, "-c", code, str(resnet), str(video), args.device],
                    timeout=args.timeout,
                )
                return output.splitlines()[-1]

            run.step("Real ResNet checkpoint inference on one frame", resnet_predict)

        port = free_local_port()
        base_url = f"http://127.0.0.1:{port}"
        api_key = f"had-system-test-{secrets.token_hex(16)}"
        runtime_dir = output_dir / "runtime"
        runtime_dir.mkdir()
        api_log_path = output_dir / "api.log"
        env = os.environ.copy()
        env.update(
            {
                "HAD_ATTRIBUTE_BACKEND": args.backend,
                "HAD_DEVICE": args.device,
                "HAD_DETECTOR_MODEL": str(yolo),
                "HAD_DATA_DIR": str(runtime_dir),
                "HAD_API_KEY": api_key,
                "HAD_ALLOW_UNAUTHENTICATED_LOCAL": "false",
                "HAD_MIN_TRACK_FRAMES": "1",
                "HAD_TARGET_FPS": "5",
            }
        )
        if resnet is not None:
            env["HAD_RESNET_CHECKPOINT"] = str(resnet)

        api_log = api_log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "silhouette_detector.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "info",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=api_log,
            stderr=subprocess.STDOUT,
        )

        job_id = ""
        try:
            run.step(
                "Start local API",
                lambda: wait_for_server(base_url, process, 45) or f"listening at {base_url}",
            )

            def negative_api_checks() -> str:
                request_json(
                    f"{base_url}/api/v1/jobs/not-a-real-job",
                    expected_status=401,
                )
                request_json(
                    f"{base_url}/api/v1/jobs/not-a-real-job",
                    headers={"X-API-Key": api_key},
                    expected_status=404,
                )
                body, boundary = multipart_bytes("file", "invalid.txt", b"not a video")
                request_json(
                    f"{base_url}/api/v1/jobs",
                    method="POST",
                    headers={
                        "X-API-Key": api_key,
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                    body=body,
                    expected_status=415,
                )
                return "401 without key, 404 unknown job, 415 invalid extension"

            run.step("API authentication and validation", negative_api_checks)

            def submit_and_wait() -> str:
                nonlocal job_id
                submitted = upload_video(port, api_key, video)
                job_id = str(submitted.get("job_id", ""))
                if not job_id:
                    raise TestFailure(f"upload response has no job_id: {submitted}")
                deadline = time.monotonic() + args.timeout
                last_progress = -1.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise TestFailure(f"API process stopped with exit code {process.returncode}")
                    status = request_json(
                        f"{base_url}/api/v1/jobs/{job_id}",
                        headers={"X-API-Key": api_key},
                        timeout=15,
                    )
                    state = status.get("state")
                    progress = float(status.get("progress", 0.0))
                    if progress != last_progress:
                        print(f"      state={state}, progress={progress:.1%}", flush=True)
                        last_progress = progress
                    if state == "completed":
                        if progress != 1.0:
                            raise TestFailure(f"completed job has progress={progress}")
                        return f"job_id={job_id}, state=completed"
                    if state == "failed":
                        raise TestFailure(f"video job failed: {status.get('error')}")
                    if state not in {"queued", "running"}:
                        raise TestFailure(f"unexpected job state: {status}")
                    time.sleep(2)
                raise TestFailure(f"job did not finish within {args.timeout} seconds")

            run.step("Complete video job through HTTP API", submit_and_wait)

            result_path = output_dir / "result.mp4"
            metadata_path = runtime_dir / "outputs" / f"{job_id}.jsonl"

            run.step(
                "Download and open result MP4",
                lambda: download_result(port, api_key, job_id, result_path)
                or validate_video(result_path),
            )
            run.step(
                "Validate JSONL, detections, tracking, and attributes",
                lambda: validate_metadata(
                    metadata_path, require_attributes=args.backend == "resnet"
                ),
            )

            def final_health() -> str:
                health = request_json(f"{base_url}/health")
                runtime = health.get("runtime")
                if not isinstance(runtime, dict) or runtime.get("initialized") is not True:
                    raise TestFailure(f"runtime was not initialized: {health}")
                if runtime.get("initialization_failed") is not False:
                    raise TestFailure(f"runtime initialization failed: {health}")
                return json.dumps(runtime, ensure_ascii=False)

            run.step("Final runtime health", final_health)
        finally:
            stop_process(process)
            api_log.close()

    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        if not isinstance(exc, TestFailure):
            traceback.print_exc()

    report_path = run.save_report(error)
    print("\n" + "=" * 72)
    if error is None:
        print("SYSTEM TEST PASSED")
        print(f"Result video: {output_dir / 'result.mp4'}")
        print(f"Report:       {report_path}")
        print(f"API log:      {output_dir / 'api.log'}")
        return 0
    print("SYSTEM TEST FAILED")
    print(f"Reason: {error}")
    print(f"Report: {report_path}")
    print(f"Artifacts and API log: {output_dir}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
