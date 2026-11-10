"""VTT syntax, independent repeated speech, and source segment validation."""
import pytest
from transcribe import parse_vtt, normalize_segments


def parse(tmp_path, body):
    path = tmp_path / 'captions.vtt'
    path.write_bytes(body.encode('utf-8'))  # Exact bytes: text mode on Windows would turn \r\n into \r\r\n.
    return parse_vtt(str(path))


def test_blocks_ids_settings_padding_and_missing_separator(tmp_path):
    out = parse(tmp_path, '\ufeffWEBVTT\r\n\r\nNOTE comment\r\n00:00.000 --> 00:01.000\r\nskip\r\n\r\nSTYLE\r\n::cue {}\r\n\r\nREGION\r\nid:one\r\n\r\ncue-id\r\n00:01.000 --> 00:02.000 align:start\r\nfirst\r\n   \r\n<c>last</c>\r\n00:03.000 --> 00:04.000\r\nnext\r\n')
    assert [s['text'] for s in out] == ['first  last', 'next']


def test_valid_long_hours_and_invalid_minutes(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n100:02:03.004 --> 100:02:04.000\nx\n\n00:60.000 --> 01:01.000\ninvalid\n')
    assert len(out) == 1 and out[0]['start'] == 360123.004


def test_overlapping_prefix_only_and_mixed_karaoke(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:01.000 --> 00:03.000\nThank you\n\n00:02.000 --> 00:04.000\nThank you all\n\n00:04.000 --> 00:05.000\nThank you all\n\n00:05.000 --> 00:06.000\nplain <00:05.100><c>spoken</c> ending &lt;literal&gt;\n')
    assert [s['text'] for s in out] == ['Thank you all', 'Thank you all', 'plain spoken ending <literal>']


@pytest.mark.parametrize('segment', [
    {'start': float('nan'), 'end': 1, 'text': 'hi'}, {'start': 2, 'end': 1, 'text': 'hi'},
    {'start': True, 'end': 1, 'text': 'hi'}, {'start': 0, 'end': 1, 'text': ' '},
    {'start': 0, 'end': float('inf'), 'text': 'hi'},
])
def test_rejects_invalid_segments(segment):
    with pytest.raises((ValueError, TypeError)):
        normalize_segments({'segments': [segment]})


def test_overlapping_separate_display_regions_are_not_merged(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000 line:10%\nHello\n\n00:01.000 --> 00:03.000 line:90%\nHello\n')
    assert len(out) == 2
    assert all(set(s) == {'start', 'end', 'text'} for s in out)


YOUTUBE_ROLLING = (
    'WEBVTT\nKind: captions\nLanguage: en\n\n'
    '00:00:00.000 --> 00:00:02.500 align:start position:0%\n \n'
    'welcome<00:00:00.500><c> to</c><00:00:01.000><c> the</c><00:00:02.000><c> keynote</c>\n\n'
    '00:00:02.500 --> 00:00:02.510 align:start position:0%\nwelcome to the keynote\n \n\n'
    '00:00:02.510 --> 00:00:05.000 align:start position:0%\nwelcome to the keynote\n'
    'today<00:00:03.000><c> we</c><00:00:04.000><c> talk</c><00:00:04.500><c> compilers</c>\n\n'
    '00:00:05.000 --> 00:00:05.010 align:start position:0%\ntoday we talk compilers\n \n\n'
    '00:00:05.010 --> 00:00:07.000 align:start position:0%\ntoday we talk compilers\n'
    'and<00:00:06.000><c> why</c>\n'
)


def test_youtube_rolling_captions_keep_each_line_once(tmp_path):
    out = parse(tmp_path, YOUTUBE_ROLLING)
    assert [s['text'] for s in out] == ['welcome to the keynote', 'today we talk compilers', 'and why']
    assert [s['start'] for s in out] == [0.0, 2.51, 5.01]
    assert out[0]['end'] == 2.51


def test_two_line_cue_without_carryover_is_kept_whole(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000\nfirst line\n\n'
                          '00:02.000 --> 00:04.000\nsecond line\nthird line\n')
    assert [s['text'] for s in out] == ['first line', 'second line third line']


def test_short_single_line_repeat_without_carryover_is_kept(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000\nyes\n\n00:03.000 --> 00:03.010\nno\n')
    assert [s['text'] for s in out] == ['yes', 'no']


def test_two_line_subtitles_keep_repeated_lines(tmp_path):
    lyrics = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000\nLet it go\nlet it go\n\n'
                             '00:02.000 --> 00:04.000\nlet it go\nCan\'t hold it back anymore\n')
    assert [s['text'] for s in lyrics] == ['Let it go let it go', "let it go Can't hold it back anymore"]
    dialogue = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000\n- Is it him?\n- Yes.\n\n'
                               '00:02.000 --> 00:04.000\n- Yes.\n- Are you sure?\n')
    assert [s['text'] for s in dialogue] == ['- Is it him? - Yes.', '- Yes. - Are you sure?']
