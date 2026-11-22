"""Shorten the walkthrough video: speed up only the marked CPU-generation stretches.

uv run --with imageio-ffmpeg python scripts/make_video.py
Reads frontend/e2e/video/walkthrough.webm + marks.json, writes walkthrough-short.webm.
The captions in those stretches already say they are sped up.
"""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

VIDEO = Path("frontend/e2e/video")


def main() -> None:
    marks = json.loads((VIDEO / "marks.json").read_text())
    speed, fast = marks["speed"], marks["fast"]
    segments, t = [], 0.0
    for start, end in fast:
        if start > t:
            segments.append((t, start, 1))
        segments.append((start, end, speed))
        t = end
    segments.append((t, None, 1))
    parts = []
    for i, (a, b, s) in enumerate(segments):
        trim = f"trim=start={a}" + (f":end={b}" if b is not None else "")
        parts.append(f"[0:v]{trim},setpts=(PTS-STARTPTS)/{s}[v{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(segments)))
    graph += f"concat=n={len(segments)}:v=1:a=0[out]"
    out = VIDEO / "walkthrough-short.webm"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(VIDEO / "walkthrough.webm"),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "1M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            str(out),
        ],
        check=True,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
