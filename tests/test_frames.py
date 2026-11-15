"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path

import frames
import pytest


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


def test_uniform_pixels_match_source_time(tmp_path):
    """A changing fill exposes the old fps half-interval label error."""
    import subprocess
    from conftest import build_cut_clip
    clip = tmp_path / 'clock.mp4'
    build_cut_clip(clip, n=12, seg=1, fps=10)
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=4)
    expected = frames.extract_at_timestamps(str(clip), tmp_path / 'truth', [f['timestamp_seconds'] for f in out])[0]
    def pixels(path):
        return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', 'scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    assert [f['timestamp_seconds'] for f in out] == [0, 3, 6, 9]
    assert all(frames._frame_delta(pixels(a['path']), pixels(b['path'])) < 2 for a, b in zip(out, expected))


def test_fractional_seek_and_cap_use_actual_span(tmp_path):
    from conftest import build_static_clip
    clip = tmp_path / 'fps.mp4'
    build_static_clip(clip, duration=6, fps='30000/1001')
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=3, start_seconds=0.151, end_seconds=100)
    assert len(out) == 3 and out[-1]['timestamp_seconds'] > 4
    assert out[0]['timestamp_seconds'] >= 0.151
    for frame in out:
        assert abs(frame['timestamp_seconds'] * 30000 / 1001 - round(frame['timestamp_seconds'] * 30000 / 1001)) < 0.01


def test_nonzero_input_pts_are_source_relative(tmp_path):
    import subprocess
    clip = tmp_path / 'offset.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x120:r=10:d=3', '-vf', 'setpts=PTS+7/TB', '-c:v', 'libx264', str(clip)], check=True)
    out = frames.extract(str(clip), tmp_path / 'f', fps=1, max_frames=3)
    assert [f['timestamp_seconds'] for f in out] == [0, 1, 2]
    # FFmpeg 7.1 muxes this shifted clip one frame (0.1s) short; either is source-relative, 10 would not be.
    assert 2.9 <= frames.get_metadata(str(clip))['duration_seconds'] <= 3


@pytest.mark.parametrize('fps', [float('nan'), float('inf'), -1, 0])
def test_invalid_fps_rejected_before_extraction(static_clip, tmp_path, fps):
    with pytest.raises(SystemExit, match='fps'):
        frames.extract(str(static_clip), tmp_path / 'f', fps=fps)


def test_gap_with_corrupt_input_still_fails(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_bytes(b'not media')
    with pytest.raises(SystemExit):
        frames.extract_keyframes(str(path), tmp_path / 'f', start_seconds=1, end_seconds=2)


def test_variable_rate_timestamps_are_actual_source_frames(tmp_path):
    import json
    import subprocess
    clip = tmp_path / 'vfr.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x120:r=30:d=6',
                    '-vf', "select='if(lt(t,3),not(mod(n,3)),not(mod(n,7)))'", *frames.sync_args(), '-c:v', 'libx264', str(clip)], check=True)
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(clip)], capture_output=True, text=True, check=True)
    source_times = [float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']]
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=4, start_seconds=0.17)
    assert len(out) == 4 and out[-1]['timestamp_seconds'] > 4
    assert all(any(abs(f['timestamp_seconds'] - t) < 0.002 for t in source_times) for f in out)


def test_single_frame_cap_keeps_first_real_frame(static_clip, tmp_path):
    out = frames.extract(str(static_clip), tmp_path / 'f', fps=2, max_frames=1)
    assert len(out) == 1 and out[0]['timestamp_seconds'] == 0


def test_uniform_candidate_count_includes_deduplicated_frames(static_clip, tmp_path):
    out, meta = frames.extract_scene_or_uniform(str(static_clip), tmp_path / 'f', fps=2, target_frames=6)
    assert meta['candidate_count'] == len(out) + meta['deduped_count']


@pytest.mark.parametrize('source_width,scaled_first', [(None, False), (1000, False), (1024, True), (1920, True)])
def test_scene_pass_scales_first_only_for_sources_twice_the_frame_width(monkeypatch, tmp_path, source_width, scaled_first):
    import subprocess
    seen = []
    real = frames.run_text
    def run(cmd, **kw):
        if '-vf' not in cmd:
            return real(cmd, **kw)
        seen.append(cmd[cmd.index('-vf') + 1])
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(frames, 'run_text', run)
    frames.extract_scene_candidates('video.mp4', tmp_path, resolution=512, source_width=source_width)
    vf = seen[0]
    assert vf.startswith('scale=') is scaled_first and vf.endswith('showinfo')


# --- Gap fill: cuts are anchors, wide gaps between them get grid frames ---

def _build_sparse_cut_clip(path: Path, cuts: int = 9, seg: float = 0.4, tail: float = 40.0, fps: int = 10) -> float:
    """``cuts`` hard colour cuts in the first ``cuts * seg`` seconds, then a
    ``testsrc`` tail that changes all the time but never cuts. Returns the
    duration."""
    import subprocess
    from conftest import COLORS
    inputs: list[str] = []
    for i in range(cuts):
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={COLORS[i % len(COLORS)]}:s=320x240:r={fps}"]
    inputs += ["-f", "lavfi", "-t", str(tail), "-i", f"testsrc=s=320x240:r={fps}"]
    n = cuts + 1
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
        "-filter_complex", "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]",
        "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return cuts * seg + tail


@pytest.fixture(scope="module")
def sparse_cut_clip(tmp_path_factory) -> tuple[Path, float]:
    path = tmp_path_factory.mktemp("clips") / "sparse.mp4"
    return path, _build_sparse_cut_clip(path)


def _grid(times):
    return [{"timestamp_seconds": t, "path": f"fill_{i:04d}.jpg", "reason": "gap-fill"} for i, t in enumerate(times)]


def test_select_fills_only_wide_gaps_and_keeps_away_from_cuts():
    grid = _grid([float(t) for t in range(0, 100, 10)])
    # Cuts at 0, 15 and 60; step 10, so only gaps wider than 20 s are filled.
    fills, gaps = frames._select_fills([0.0, 15.0, 60.0], grid, 100.0, 10.0, budget=50)
    assert gaps == [(15.0, 60.0), (60.0, 100.0)]
    # 20..50 inside the first gap (not 60: within half a step of the cut), 70..90 after the last cut.
    assert [f["timestamp_seconds"] for f in fills] == [20.0, 30.0, 40.0, 50.0, 70.0, 80.0, 90.0]


def test_select_fills_thins_evenly_to_the_budget_and_handles_no_budget():
    grid = _grid([float(t) for t in range(0, 100, 10)])
    fills, gaps = frames._select_fills([0.0], grid, 100.0, 10.0, budget=3)
    assert [f["timestamp_seconds"] for f in fills] == [10.0, 50.0, 90.0]
    assert gaps == [(0.0, 100.0)]
    assert frames._select_fills([0.0], grid, 100.0, 10.0, budget=0) == ([], [])
    # Cuts no further apart than two steps leave nothing to fill.
    assert frames._select_fills([0.0, 20.0, 40.0, 60.0, 80.0], grid, 100.0, 10.0, budget=50) == ([], [])


def test_scene_gap_fill_covers_a_slow_tail(sparse_cut_clip, tmp_path):
    """The brief's case: 9 cuts in 3.6 s, then a 40 s tail with no cuts, budget
    40. Before, this returned 9 frames and none after 3.6 s."""
    clip, duration = sparse_cut_clip
    out, meta = frames.extract_scene_or_uniform(str(clip), tmp_path / "f", fps=2.0, target_frames=40, max_frames=100)
    assert meta["engine"] == "scene" and meta["fallback"] is False
    assert meta["scene_count"] == 9 and meta["fill_count"] > 20
    assert meta["candidate_count"] == meta["scene_count"] + meta["fill_count"]
    assert meta["candidate_count"] == len(out) + meta["deduped_count"]
    assert len(meta["filled_gaps"]) == 1 and meta["filled_gaps"][0][1] == pytest.approx(duration)
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts) and len(out) <= 40
    assert max(b - a for a, b in zip(ts, ts[1:])) < 3.0
    assert {fr["reason"] for fr in out} >= {"scene-change", "gap-fill"}
    # One contiguous frame_*.jpg sequence (so dedup ran), no stray fill files.
    folder = tmp_path / "f"
    assert sorted(fr["path"] for fr in out) == sorted(str(p) for p in folder.glob("frame_*.jpg"))
    assert not list(folder.glob("fill_*.jpg")) and not list(folder.glob(".renumber_*"))
    assert [fr["index"] for fr in out] == list(range(len(out)))


def test_scene_gap_fill_stays_in_the_focus_window_and_under_the_cap(sparse_cut_clip, tmp_path):
    clip, _ = sparse_cut_clip
    out, meta = frames.extract_scene_or_uniform(
        str(clip), tmp_path / "f", fps=2.0, target_frames=30, max_frames=12, start_seconds=0.5, end_seconds=20.0)
    assert meta["engine"] == "scene" and meta["fill_count"] > 0
    assert len(out) <= 12
    assert all(0.5 <= fr["timestamp_seconds"] <= 20.0 for fr in out)
    assert any(fr["timestamp_seconds"] > 10.0 for fr in out)


def test_scene_no_fill_when_cuts_already_cover_the_clip(cut_clip, tmp_path):
    out, meta = frames.extract_scene_or_uniform(str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100)
    assert meta["fill_count"] == 0 and meta["filled_gaps"] == []
    assert "gap-fill" not in {fr["reason"] for fr in out}
    assert not list((tmp_path / "f").glob("fill_*.jpg"))


def test_static_fallback_leaves_no_fill_files(static_clip, tmp_path):
    out, meta = frames.extract_scene_or_uniform(str(static_clip), tmp_path / "f", fps=2.0, target_frames=6)
    assert meta["engine"] == "uniform"
    assert not list((tmp_path / "f").glob("fill_*.jpg"))


@pytest.mark.parametrize('source_width,scaled_first', [(None, False), (1920, True)])
def test_fill_grid_shares_the_scene_decode_and_scale(monkeypatch, tmp_path, source_width, scaled_first):
    import subprocess
    seen = []
    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(frames, 'run_text', run)
    monkeypatch.setattr(frames, 'sync_args', lambda: ('-fps_mode', 'vfr'))
    frames.extract_scene_candidates('video.mp4', tmp_path, resolution=512, source_width=source_width, fill_rate=0.5)
    assert len(seen) == 1  # one ffmpeg run writes both the cuts and the grid
    graph = seen[0][seen[0].index('-filter_complex') + 1]
    assert graph.startswith('scale=') is scaled_first
    assert graph.count('scale=') == (1 if scaled_first else 2)
    assert 'showinfo@scene' in graph and 'showinfo@fill' in graph


def test_renumber_sequence_merges_fill_and_frame_files_in_time_order(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    for name in ("frame_0001.jpg", "frame_0002.jpg", "fill_0001.jpg"):
        (d / name).write_bytes(name.encode())
    merged = frames._renumber_sequence([
        {"path": str(d / "frame_0001.jpg"), "timestamp_seconds": 0.0, "index": 0},
        {"path": str(d / "frame_0002.jpg"), "timestamp_seconds": 4.0, "index": 1},
        {"path": str(d / "fill_0001.jpg"), "timestamp_seconds": 2.0, "index": 0},
    ], d)
    assert [Path(m["path"]).name for m in merged] == ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]
    assert [m["index"] for m in merged] == [0, 1, 2]
    assert (d / "frame_0002.jpg").read_bytes() == b"fill_0001.jpg"
    assert sorted(p.name for p in d.iterdir()) == ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]


def test_emitted_frames_reads_one_named_showinfo():
    stderr = ("[showinfo@scene @ 0x1] n:   0 pts:      0 pts_time:0\n"
              "[showinfo@fill @ 0x2] n:   0 pts:      0 pts_time:0\n"
              "[showinfo@fill @ 0x2] n:   1 pts:     10 pts_time:2.5\n")
    out = frames._emitted_frames([Path('a.jpg'), Path('b.jpg')], stderr, 1.0, 'gap-fill', 'fill')
    assert [f["timestamp_seconds"] for f in out] == [1.0, 3.5]
