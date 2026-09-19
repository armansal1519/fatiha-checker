"""Pure harness checks; actual model/transport evidence comes from opt-in scripts."""
import struct

import numpy as np

from benchmark_websocket import number_summary, pcm_frame, state_difference


def test_frame_preserves_sequence_offset_and_pcm_boundary_values():
    frame = pcm_frame(17, 2048, np.array([-2.0, -1.0, 0, .5, 1, 2], dtype=np.float32))
    assert struct.unpack("<II", frame[:8]) == (17, 2048)
    assert np.frombuffer(frame[8:], dtype="<i2").tolist() == [-32767, -32767, 0, 16383, 32767, 32767]


def test_unmatched_recordings_are_not_counted_as_zero_latency():
    assert number_summary([None, 3, 1, None])["median"] == 2
    assert number_summary([None])["count"] == 0
    assert number_summary([None])["median"] is None


def test_final_state_diff_reports_missing_words_and_revised_colors():
    left = [{"index": 9, "state": "match"}, {"index": 10, "state": "difference"}]
    right = [{"index": 2, "state": "match"}, {"index": 9, "state": "difference"}]
    assert state_difference(left, right) == [
        {"index": 2, "left": None, "right": "match"},
        {"index": 9, "left": "match", "right": "difference"},
        {"index": 10, "left": "difference", "right": None},
    ]
