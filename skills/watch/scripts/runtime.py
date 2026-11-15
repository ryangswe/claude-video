"""Small cross-platform process and stream helpers; no optional dependencies."""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import date


def configure_stdio() -> None:
    """Call at script entry only, leaving imported modules' streams untouched."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(encoding='utf-8', errors='backslashreplace')
            except (OSError, ValueError):
                pass


def diagnostic(value, limit: int = 3000) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value or '').strip()[-limit:]


def run_text(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Capture diagnostics predictably, including Windows pipes and launch failures."""
    try:
        return subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f'{cmd[0]} timed out: {diagnostic(exc.stderr)}') from None
    except OSError as exc:
        raise SystemExit(
            f'Cannot run {cmd[0]} ({type(exc).__name__}). Check that the executable is installed '
            'and permitted by your operating system, then retry.'
        ) from None


# yt-dlp versions are release dates. Sites such as YouTube change often, and an old release
# is the usual cause of a 403 or a missing format. Upstream PR #227 measured a 46-day gap
# between two healthy releases and a 76-day-old binary that had started failing.
YT_DLP_STALE_DAYS = 60
YT_DLP_UPDATE = 'brew upgrade yt-dlp, pipx upgrade yt-dlp, or winget upgrade --id yt-dlp.yt-dlp --exact'


def ytdlp_release_age(version: str | None, today: date | None = None) -> int | None:
    """Days since a ``YYYY.MM.DD`` yt-dlp version was released; None for other schemes."""
    match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2})\b', (version or '').strip())
    if not match:
        return None
    try:
        released = date(*map(int, match.groups()))
    except ValueError:
        return None
    return max(0, ((today or date.today()) - released).days)


def ytdlp_age_note(version: str | None, today: date | None = None) -> str:
    """One sentence on the installed yt-dlp's age, with an update hint when it is stale."""
    days = ytdlp_release_age(version, today)
    if days is None:
        return ''
    note = f'Installed yt-dlp {version.split()[0]} was released {days} days ago.'
    if days > YT_DLP_STALE_DAYS:
        note += f' Releases this old often fail on YouTube; update it first ({YT_DLP_UPDATE}).'
    return note
