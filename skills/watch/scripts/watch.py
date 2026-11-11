#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Writes a markdown report (frame paths + transcript) to report.md in the work dir.
A report that fits in one tool result is printed whole; a larger one is printed
as a short version that points to a frame list and transcript parts on disk.
The agent then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import ConfigError, frame_cap, get_config, load_gemini_key, resolve_engine  # noqa: E402
import gemini  # noqa: E402
from download import download, fetch_captions, is_url, auth_args, source_width  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract_at_timestamps, extract_keyframes, extract_scene_or_uniform, format_time, get_metadata, merge_frames, parse_time, parse_timestamps, validate_controls  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402
from runtime import configure_stdio  # noqa: E402


LOCAL_ONLY_FLAGS = (("--detail", "detail"), ("--fps", "fps"), ("--max-frames", "max_frames"),
                    ("--timestamps", "timestamps"), ("--whisper", "whisper"), ("--sub-lang", "sub_lang"))

# Hosts cut long tool output: Claude Code keeps only a 2 KB preview of output past
# about 25 KB. Reports above this size are printed as a short version instead.
STDOUT_LIMIT = 20_000
# Each transcript part stays small enough for a single Read call.
TRANSCRIPT_PART_LIMIT = 40_000
# A longer frame list goes to frames.md rather than into the printed report.
FRAME_LIST_LIMIT = 12_000
FRAME_ORDER_NOTE = ("Frames are in chronological order; `t=MM:SS` is the absolute timestamp "
                    "in the source video.")


def utf8_size(text: str) -> int:
    return len(text.encode("utf-8", "backslashreplace"))


def write_report_file(path: Path, text: str) -> None:
    """LF endings on every OS (sizes stay as measured) and no crash on undecodable paths."""
    with open(path, "w", encoding="utf-8", errors="backslashreplace", newline="\n") as handle:
        handle.write(text)


def _frame_entry(frame: dict, name: str) -> str:
    return (f"- `{name}` (t={format_time(frame['timestamp_seconds'])}, "
            f"reason={frame.get('reason', 'selected')})")


def frame_lines(frames: list[dict], work: Path, compact: bool = False) -> list[str]:
    """The report's Frames section. Compact mode names files relative to the frames folder
    and moves a long list to frames.md, so the printed report stays small."""
    if not frames:
        return ["## Frames", "", "_No frames extracted._"]
    folder = work / "frames"
    lines = ["## Frames", "", f"Frames live at: `{folder}`", ""]
    full_entries = [_frame_entry(frame, frame["path"]) for frame in frames]
    if not compact:
        return lines + [f"**Read each frame path below with the Read tool to view the image.** {FRAME_ORDER_NOTE}",
                        "", *full_entries]
    short_entries = [_frame_entry(frame, Path(frame["path"]).name if Path(frame["path"]).parent == folder
                                  else frame["path"]) for frame in frames]
    if utf8_size("\n".join(short_entries)) <= FRAME_LIST_LIMIT:
        return lines + [f"**Read each frame below with your image-viewing tool, at `<frames folder>/<file name>`.** "
                        f"{FRAME_ORDER_NOTE}", "", *short_entries]
    listing = work / "frames.md"
    write_report_file(listing, "\n".join(full_entries) + "\n")
    return lines + [f"**{len(frames)} frames are listed with full paths in `{listing}`.** Read that list, then "
                    f"read each frame with your image-viewing tool. {FRAME_ORDER_NOTE}"]


def write_transcript_parts(segments: list[dict], folder: Path, note: str,
                           limit: int = TRANSCRIPT_PART_LIMIT) -> list[dict]:
    """Split the transcript at segment boundaries into files of at most ``limit`` bytes,
    counting the note and code fence around each part (a single segment longer than that
    gets a part of its own)."""
    wrapper = utf8_size(f"{note} Part 999 of 999.\n\n```\n\n```\n")
    groups, current, size = [], [], wrapper
    for segment in segments:
        added = utf8_size(format_transcript([segment])) + 1
        if current and size + added > limit:
            groups.append(current)
            current, size = [], wrapper
        current.append(segment)
        size += added
    if current:
        groups.append(current)
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for number, group in enumerate(groups, 1):
        path = folder / f"part-{number:02d}.md"
        write_report_file(path, f"{note} Part {number} of {len(groups)}.\n\n```\n{format_transcript(group)}\n```\n")
        parts.append({"path": path, "start": group[0]["start"], "end": group[-1]["end"]})
    return parts


def run_gemini(args, config, key, start_sec, end_sec, auth) -> int:
    """Google watches the video. No captions, frames, or Whisper; no fallback on failure."""
    ignored = [flag for flag, name in LOCAL_ONLY_FLAGS if getattr(args, name) is not None]
    ignored += [flag for flag, on in (("--no-whisper", args.no_whisper), ("--no-dedup", args.no_dedup)) if on]
    clip = (start_sec, end_sec) if start_sec is not None or end_sec is not None else None
    uploaded, warning, result, error, sent, work = None, None, None, None, "URL sent to Google", None
    try:
        if gemini.is_youtube(args.source):
            video = {"uri": args.source}
        else:
            parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
            if parent:
                parent.mkdir(parents=True, exist_ok=True)
            sent = "not sent to Google"
            work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
            print("[watch] downloading media…" if is_url(args.source) else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", **(auth if is_url(args.source) else {}))
            sent = "upload to Google failed"
            print("[watch] uploading to the Gemini Files API…", file=sys.stderr)
            uploaded = gemini.upload_file(Path(media["video_path"]), key)
            video = {"uri": uploaded["uri"], "mime_type": uploaded["mime_type"]}
            sent = "video uploaded to Google, deleted after the answer"
        print(f"[watch] asking {config['gemini_model']}…", file=sys.stderr)
        result = gemini.ask(video, args.question, model=config["gemini_model"], key=key,
                            clip=clip, timeout=config["gemini_timeout"])
    except SystemExit as exc:
        error = str(exc)
    finally:
        if uploaded:
            warning = gemini.delete_file(uploaded["name"], key)
        if work:  # Run-owned; holds at most a downloaded copy. A local source file lives elsewhere.
            shutil.rmtree(work, ignore_errors=True)

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source} ({sent})")
    print(f"- **Engine:** {config['gemini_model']} ({result['processing'] if result else 'failed'})")
    if clip:
        print(f"- **Focus range:** {format_time(start_sec or 0)} → {format_time(end_sec) if end_sec is not None else 'end'}")
    if ignored:
        print(f"- **Ignored local options:** {', '.join(ignored)} (these only apply with --engine local)")
    if result and result["total_tokens"] is not None:
        print(f"- **Gemini tokens:** {result['total_tokens']}")
    if warning:
        print(f"- **Cleanup warning:** {warning}")
    print()
    if error:
        print("## Unavailable evidence")
        print()
        print(f"- {error}")
        return 1
    print("## Answer (from Gemini)")
    print()
    print("_These are Gemini's observations of the video, not frames you viewed yourself. "
          "Relay them as such; rerun with `--engine local` to inspect frames directly._")
    print()
    print(result["text"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=None, help="Override frame cap")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument(
        "--detail",
        choices=["transcript", "efficient", "balanced", "talk", "token-burner"],
        default=None,
        help="Fidelity/speed dial: transcript (no frames), efficient (fast keyframes, cap 50), "
             "balanced (scene, cap 100), talk (scene, cap 300: every slide of a long talk), "
             "token-burner (scene, uncapped).",
    )
    ap.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Comma-separated absolute timestamps (SS, MM:SS, HH:MM:SS) to grab a frame at, "
             "e.g. transcript-flagged 'look here' moments. Added on top of the detail frames "
             "(reserved against the cap); with --detail transcript these become the only frames.",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable all local/cloud transcription fallbacks; native captions still work.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai", "whisperx", "whispercpp"],
        default=None,
        help="Select the fallback backend for this run; native captions still come first.",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate frame removal. Keeps visually identical "
             "frames (static screen recordings, held slides) instead of collapsing them.",
    )
    ap.add_argument("--sub-lang", default=None, help="Exact caption language preference (default auto/native)")
    cookies = ap.add_mutually_exclusive_group()
    cookies.add_argument("--cookies", default=None, help="Explicit cookie file (yt-dlp may update this jar)")
    cookies.add_argument("--cookies-from-browser", default=None, help="Explicit yt-dlp browser selector")
    ap.add_argument("--engine", choices=["auto", "gemini", "local"], default=None,
                    help="gemini: Google watches the video and answers (needs GEMINI_API_KEY). "
                         "local: frames + transcript on this machine. Default auto: gemini when a key exists.")
    ap.add_argument("--question", default=None,
                    help="The user's question. Sent to Gemini on a gemini run; unused by the local engine.")
    args = ap.parse_args()
    if args.no_whisper and args.whisper:
        ap.error("--no-whisper conflicts with --whisper")

    config = get_config(backend_override="none" if args.no_whisper else args.whisper)
    detail = args.detail or config["detail"]
    max_frames = args.max_frames if args.max_frames is not None else frame_cap(detail)
    budget_cap = max_frames if max_frames is not None else 100
    start_sec, end_sec = parse_time(args.start), parse_time(args.end)
    validate_controls(args.resolution, max_frames, start_sec, end_sec, args.fps)
    cue_timestamps = parse_timestamps(args.timestamps)
    backend_choice = "none" if args.no_whisper else args.whisper or config["whisper_backend"]
    cookies_file = args.cookies if args.cookies is not None else (None if args.cookies_from_browser else config["cookies_file"])
    cookies_browser = args.cookies_from_browser if args.cookies_from_browser is not None else (None if args.cookies else config["cookies_from_browser"])
    auth_args(cookies_file, cookies_browser)  # Validate even before starting network work.
    auth = {"cookies_file": cookies_file, "cookies_from_browser": cookies_browser}

    gemini_key = load_gemini_key()
    if resolve_engine(args.engine or config["engine"], bool(gemini_key)) == "gemini":
        return run_gemini(args, config, gemini_key, start_sec, end_sec, auth)

    parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
    print(f"[watch] working dir: {work}", file=sys.stderr)
    url_source = is_url(args.source)
    dl = {"subtitle_path": None, "info": {}, "downloaded": False}
    errors = []
    all_segments = []
    track_available = False
    transcript_source = None
    video_path = None
    visual_error = None
    gaps = []

    if url_source:
        print("[watch] checking metadata/captions via yt-dlp…", file=sys.stderr)
        dl = fetch_captions(args.source, work / "download", sub_lang=args.sub_lang or config["sub_lang"], **auth)
        errors.extend(dl.get("errors", []))
        if dl.get("subtitle_path"):
            try:
                all_segments = parse_vtt(dl["subtitle_path"])
                track_available = bool(all_segments)
                track = dl.get("caption_track") or {}
                transcript_source = (f"captions ({track.get('language', 'unknown')}, "
                                     f"{track.get('kind', 'unknown')}, {track.get('provenance', 'unknown')})")
            except (OSError, ValueError) as exc:
                errors.append(f"Caption parsing failed: {exc}")

    audio_only = detail == "transcript" and not cue_timestamps
    # Captions are assessed before range filtering: a quiet interval is not a missing track.
    need_media = not (audio_only and (track_available or backend_choice == "none") and url_source)
    if need_media:
        try:
            print("[watch] downloading media…" if url_source else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", audio_only=audio_only,
                             **({"context": dl, "max_width": source_width(args.resolution), **auth}
                                if url_source else {}))
            dl.update(media)
            video_path = dl["video_path"]
        except SystemExit as exc:
            visual_error = f"Media unavailable: {exc}"
            errors.append(visual_error)

    try:
        duration = float((dl.get("info") or {}).get("duration") or 0)
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
    except (TypeError, ValueError):
        duration = 0.0
    meta = {"duration_seconds": duration, "width": None, "height": None, "codec": None,
            "has_audio": False, "has_video": False}
    if video_path:
        try:
            meta = get_metadata(video_path)
        except SystemExit as exc:
            visual_error = f"Visuals/audio metadata unavailable: {exc}"
            errors.append(visual_error)
    full_duration = meta["duration_seconds"]
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")
    effective_start = start_sec or 0.0
    effective_end = min(end_sec, full_duration) if end_sec is not None and full_duration > 0 else end_sec or full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=budget_cap)
    else:
        fps, target = auto_fps(effective_duration, max_frames=budget_cap)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    frames, cue_frames = [], []
    frame_meta = {"engine": "none", "candidate_count": 0, "selected_count": 0, "fallback": False}
    cue_meta = {}
    detail_budget = max_frames
    if video_path and meta.get("has_video", bool(meta.get("width"))):
        try:
            if cue_timestamps:
                cue_frames, cue_meta = extract_at_timestamps(
                    video_path, work / "frames", cue_timestamps, resolution=args.resolution,
                    max_frames=max_frames, start_seconds=start_sec, end_seconds=effective_end or end_sec)
            detail_budget = None if max_frames is None else max_frames - len(cue_frames)
            if detail != "transcript" and detail_budget != 0:
                kwargs = dict(resolution=args.resolution, max_frames=detail_budget,
                              start_seconds=start_sec, end_seconds=effective_end or end_sec, dedup=not args.no_dedup)
                if detail == "efficient":
                    frames, frame_meta = extract_keyframes(video_path, work / "frames", **kwargs)
                else:
                    frames, frame_meta = extract_scene_or_uniform(video_path, work / "frames", fps=fps, target_frames=target,
                                                                  source_width=meta.get("width"), **kwargs)
        except SystemExit as exc:
            visual_error = f"Visual extraction unavailable: {exc}"
            errors.append(visual_error)
    elif video_path and not visual_error and (detail != "transcript" or cue_timestamps):
        visual_error = "No video stream; visual evidence is unavailable."
    if cue_frames:
        frames = merge_frames(frames, cue_frames)

    transcript_state = "missing"
    if not track_available and backend_choice != "none" and video_path and meta.get("has_audio"):
        backend, api_key = ((backend_choice, None) if backend_choice in ("whisperx", "whispercpp")
                            else load_api_key(None if backend_choice == "auto" else backend_choice))
        if backend:
            try:
                all_segments, used_backend = transcribe_video(video_path, work / "audio.mp3", backend=backend, api_key=api_key)
                gaps = getattr(all_segments, "gaps", [])
                track_available = True
                transcript_state = "no speech" if not all_segments else "available"
                if used_backend == "whisperx":
                    language = config["whisperx_language"] or "auto (unverified)"
                    transcript_source = f"whisper (whisperx {config['whisperx_model']}, language {language})"
                elif used_backend == "whispercpp":
                    language = config["whispercpp_language"] or getattr(all_segments, "language", None) or "auto"
                    transcript_source = f"whisper (whisper.cpp {config['whispercpp_model']}, language {language})"
                else:
                    transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                transcript_state = "failed"
                errors.append(f"Transcription failed: {exc}")
        else:
            transcript_state = f"unavailable: no matching API key for {backend_choice}"
    elif not track_available and backend_choice == "none":
        transcript_state = "fallback disabled; no captions available"
    elif not track_available and video_path and not meta.get("has_audio"):
        transcript_state = "no audio stream" if not visual_error else "audio metadata unavailable"

    transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
    transcript_text = format_transcript(transcript_segments)
    if track_available:
        transcript_state = "available" if transcript_segments else ("no speech in selected range" if focused and all_segments else "no speech")
    for error in errors:
        print(f"[watch] {error}", file=sys.stderr)

    info = dl.get("info") or {}

    with contextlib.redirect_stdout(io.StringIO()) as header_buf:
        print()
        print("# watch: video report")
        print()
        print(f"- **Source:** {args.source}")
        if video_path:
            print(f"- **Local media:** `{video_path}`")
        if visual_error:
            print(f"- **Visual status:** {visual_error}")
        if errors:
            print("- **Result:** partial evidence" if frames or transcript_segments else "- **Result:** unavailable evidence")
        if info.get("title"):
            print(f"- **Title:** {info['title']}")
        if info.get("uploader"):
            print(f"- **Uploader:** {info['uploader']}")
        print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
        if focused:
            print(
                f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
                f"({effective_duration:.1f}s)"
            )
        if meta.get("width") and meta.get("height"):
            print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
        range_mode = "focused" if focused else "full"
        print(f"- **Detail:** {detail}")
        detail_count = frame_meta.get("selected_count", 0)
        if detail != "transcript":
            cap_label = "unlimited" if detail_budget is None else str(detail_budget)
            engine = frame_meta.get("engine", "scene")
            fallback = " fallback" if frame_meta.get("fallback") else ""
            deduped = frame_meta.get("deduped_count", 0)
            dedup_note = f", {deduped} near-duplicate{'s' if deduped != 1 else ''} dropped" if deduped else ""
            print(
                f"- **Frames:** {detail_count} selected from {frame_meta.get('candidate_count', detail_count)} "
                f"candidates ({engine}{fallback}{dedup_note}, {range_mode} range, budget {target}, cap {cap_label})"
            )
        elif not cue_frames:
            print("- **Frames:** skipped (transcript detail)")
        if cue_frames:
            dropped = cue_meta.get("dropped_out_of_window", 0)
            drop_note = f", {dropped} dropped outside range" if dropped else ""
            print(
                f"- **Cue frames:** {len(cue_frames)} at transcript-flagged timestamps "
                f"(transcript-cue{drop_note})"
            )
        if frames:
            print(f"- **Frame size:** max {args.resolution}px wide, max 1998px tall")
        if transcript_segments:
            in_range = " in range" if focused else ""
            print(
                f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
                f"(via {transcript_source or 'captions'})"
            )
        else:
            print(f"- **Transcript:** {transcript_state}")
        if gaps:
            print("- **Transcript status:** partial; missing intervals:")
            for gap in gaps:
                end = gap["end"] if gap["end"] is not None else full_duration
                print(f"  - {format_time(gap['start'])} → {format_time(end)}")

        if detail == "token-burner" and len(frames) > 250:
            print()
            print(
                f"> **Warning:** token-burner detail selected {len(frames)} frames. "
                "This may use a large number of image tokens."
            )

        kept_after_dedup = frame_meta.get("candidate_count", 0) - frame_meta.get("deduped_count", 0)
        thinned = kept_after_dedup > frame_meta.get("selected_count", 0)
        sparse = detail in ("efficient", "balanced") or (detail == "talk" and thinned)
        if not focused and full_duration > 600 and sparse:
            mins = int(full_duration // 60)
            print()
            print(
                f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length "
                f"under `{detail}` detail — its cap spreads thin across the full clip. For better results, "
                "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a section, "
                + ("" if detail == "talk" else
                   "use `--detail talk` to keep up to 300 scene-change frames (every slide of most talks), ")
                + "or use `--detail token-burner` to keep every scene-change frame across the whole video."
            )

    header = header_buf.getvalue().rstrip("\n")
    missing = []
    source_note = None
    if transcript_text:
        label = transcript_source or "captions"
        source_note = (f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_"
                       if focused else f"_Source: {label}._")
        write_report_file(work / "transcript.md", f"{source_note}\n\n{transcript_text}\n")
    else:
        missing = [f"_{transcript_state.capitalize()}._"]
        if detail == "transcript" and not track_available:
            missing.append("_Re-run with `--detail balanced` for visual evidence, or configure a transcription backend with setup.py._")
    tail = []
    if errors:
        tail += ["", "## Unavailable evidence", *(f"- {error}" for error in errors)]
    tail += ["", "---", f"_Work dir: `{work}` — delete when done._"]

    full = [header, "", *frame_lines(frames, work), "", "## Transcript", ""]
    full += [source_note, "", "```", transcript_text, "```"] if transcript_text else missing
    full_text = "\n".join(full + tail) + "\n"
    report_path = work / "report.md"
    write_report_file(report_path, full_text)
    if utf8_size(full_text) <= STDOUT_LIMIT:
        print(full_text, end="")
    else:
        short = [header, "", f"> **Note:** The full report is {utf8_size(full_text) // 1024} KB, too large for one tool "
                 f"result, so it is saved at `{report_path}` and this shorter version points to files instead. "
                 "Read the frames and every transcript part it lists; open report.md or transcript.md "
                 "only to search them.", "", *frame_lines(frames, work, compact=True), "", "## Transcript", ""]
        if transcript_text:
            parts = write_transcript_parts(transcript_segments, work / "transcript", source_note)
            short += [source_note, "",
                      f"The transcript is {utf8_size(transcript_text) // 1024} KB, so it is split into {len(parts)} "
                      "parts. Read every part, in order, with your file-reading tool:", ""]
            short += [f"- `{part['path']}` ({format_time(part['start'])} → {format_time(part['end'])})" for part in parts]
        else:
            short += missing
        print("\n".join(short + tail))

    return 1 if errors and not frames and not transcript_segments and not track_available else 0


if __name__ == "__main__":
    configure_stdio()
    try:
        raise SystemExit(main())
    except (ConfigError, OSError) as exc:
        raise SystemExit(str(exc)) from None
