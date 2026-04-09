from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.benchmark import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hf_import_probe.json"
    markdown_path = output_dir / "hf_import_probe.md"

    payload = probe_hf_import_runtime(timeout_seconds=args.timeout_seconds)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote HF import probe to {json_path}")
    print(f"Wrote HF import probe markdown to {markdown_path}")


def probe_hf_import_runtime(timeout_seconds: int) -> dict[str, object]:
    start = time.perf_counter()
    command = [
        sys.executable,
        "-c",
        (
            "import json\n"
            "import time\n"
            "started = time.perf_counter()\n"
            "payload = {'ok': False}\n"
            "import torch\n"
            "payload['torch_ms'] = int((time.perf_counter() - started) * 1000)\n"
            "import transformers\n"
            "payload['transformers_ms'] = int((time.perf_counter() - started) * 1000)\n"
            "payload['ok'] = True\n"
            "print(json.dumps(payload))\n"
        ),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "ok": False,
            "error": "timeout",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "timeout_seconds": timeout_seconds,
            "returncode": process.returncode,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
        }

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    payload: dict[str, object]
    if process.returncode == 0 and stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"ok": False, "parse_error": stdout}
    else:
        payload = {"ok": False}
    payload.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": timeout_seconds,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    )
    return payload


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# HF Import Probe",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- ok: `{payload.get('ok')}`",
        f"- elapsed_ms: `{payload.get('elapsed_ms')}`",
        f"- torch_ms: `{payload.get('torch_ms')}`",
        f"- transformers_ms: `{payload.get('transformers_ms')}`",
        f"- timeout_seconds: `{payload.get('timeout_seconds')}`",
        f"- returncode: `{payload.get('returncode')}`",
    ]
    stderr = str(payload.get("stderr") or "").strip()
    if stderr:
        lines.append(f"- stderr: `{stderr[:300]}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
