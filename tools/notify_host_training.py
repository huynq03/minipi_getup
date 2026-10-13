"""Send latest HoST training status to ntfy every 30 minutes; never starts training."""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs/host_staged_recovery"


def snapshot():
  logs = list(LOGDIR.glob("train_*.log"))
  now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S")
  lines = [f"Rev13 | {now} ICT"]
  if not logs:
    return "\n".join(lines + ["No training log found."])
  log = max(logs, key=lambda p: p.stat().st_mtime)
  with log.open("rb") as stream:
    stream.seek(max(0, log.stat().st_size - 262144))
    text = stream.read().decode("utf-8", errors="replace")
  text = re.sub(r"\x1b\[[0-9;]*m", "", text)
  processes = []
  for proc in Path("/proc").glob("[0-9]*"):
    try:
      args = (proc / "cmdline").read_bytes().split(b"\0")
      if (
        b"minipi_getup.host_staged_recovery.train" in args
        and (proc / "cwd").resolve() == ROOT
      ):
        processes.append(proc.name)
    except (OSError, RuntimeError):
      continue
  status = "RUNNING" if processes else "STOPPED"
  if not processes and (
    "Traceback (most recent call last)" in text or "out of memory" in text
  ):
    status = "ERROR / STOPPED"
  age = int(time.time() - log.stat().st_mtime)
  lines += [
    f"Status: {status}; PID: {', '.join(processes) or '-'}",
    f"Run log: {log.name}",
  ]
  iterations = list(re.finditer(r"Learning iteration\s+(\d+)/(\d+)", text))
  if iterations:
    match = iterations[-1]
    current, total = map(int, match.groups())
    lines.append(f"Iteration: {current}/{total} ({current / total:.1%})")
    # Use the last fully logged iteration rather than mixing metrics from a partial block.
    block = text[match.end() :]
    if "ETA:" not in block and len(iterations) > 1:
      block = text[iterations[-2].end() : match.start()]
    for name in (
      "Mean reward",
      "Iteration time",
      "ETA",
      "Episode_end/fallen_success",
      "Episode/action_scale",
    ):
      values = re.findall(rf"{re.escape(name)}:\s*([^\n]+)", block)
      if values:
        lines.append(f"{name}: {values[-1].strip()}")
  errors = [
    line.strip()
    for line in text.splitlines()
    if "out of memory" in line or line.startswith("RuntimeError:")
  ]
  if errors:
    lines.append(f"Last error: {errors[-1][:250]}")
  lines.append(f"Log updated {age}s ago")
  try:
    gpu = subprocess.check_output(
      [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
      ],
      text=True,
      timeout=10,
      stderr=subprocess.DEVNULL,
    ).strip()
    lines.append(f"GPU (used, total, load, temp): {gpu}")
  except (OSError, subprocess.SubprocessError):
    lines.append("GPU status unavailable")
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--topic", default="HuyCodex")
  parser.add_argument("--server", default="https://ntfy.sh")
  parser.add_argument("--interval", type=int, default=1800)
  parser.add_argument("--once", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if args.interval < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.topic):
    parser.error("positive interval and valid ntfy topic required")
  while True:
    message = snapshot()
    print(message, flush=True)
    if args.dry_run:
      return
    request = urllib.request.Request(
      f"{args.server.rstrip('/')}/{args.topic}",
      data=message.encode(),
      headers={
        "Title": "Rev13 training status",
        "Content-Type": "text/plain; charset=utf-8",
      },
      method="POST",
    )
    try:
      with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
      print(f"ntfy delivered: id={result.get('id')}", flush=True)
    except Exception as exc:
      print(f"ntfy delivery failed: {exc}", flush=True)
      if args.once:
        raise
    if args.once:
      return
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
