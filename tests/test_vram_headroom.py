from __future__ import annotations

import pytest

from tools.bench.measure_vram_headroom import MEDIA_CASES, parse_sample, recommend


def test_parse_sample_reads_nvidia_smi_csv() -> None:
    sample = parse_sample("1, 812, 15498, 16311\n", 2.5)
    assert sample is not None
    assert (sample.gpu, sample.free_mib) == (1, 812)
    assert (sample.used_mib, sample.total_mib) == (15498, 16311)
    assert sample.seconds == 2.5
    assert parse_sample("[N/A], 1, 2, 3", 0.0) is None
    assert parse_sample("0, 1, 2", 0.0) is None


def test_recommend_moves_slack_above_safety_from_headroom_to_kv() -> None:
    result = recommend({0: 900, 1: 780}, current_headroom_mib=1024, safety_mib=256)
    assert result.fullest_gpu == 1
    assert result.slack_mib == 524
    assert result.headroom_mib == 500


def test_recommend_raises_headroom_when_below_safety() -> None:
    result = recommend({0: 100, 1: 400}, current_headroom_mib=512, safety_mib=256)
    assert result.fullest_gpu == 0
    assert result.slack_mib == -156
    assert result.headroom_mib == 668
    assert "raise the headroom" in result.explanation


def test_recommend_can_reach_zero_headroom() -> None:
    result = recommend({0: 1280}, current_headroom_mib=1024, safety_mib=256)
    assert result.headroom_mib == 0


def test_recommend_refuses_free_memory_beyond_the_headroom() -> None:
    # --kv-capacity auto leaves only the headroom free; more means a mismatched setup.
    result = recommend({0: 3000}, current_headroom_mib=1024, safety_mib=256)
    assert result.headroom_mib is None
    assert "--headroom-mib matches the server" in result.explanation


def test_recommend_needs_samples() -> None:
    with pytest.raises(ValueError):
        recommend({}, current_headroom_mib=1024, safety_mib=256)


def test_media_cases_are_the_cases_that_send_images_or_video() -> None:
    assert {
        "many-image-28",
        "two-heavy-media-arrivals",
        "mixed-four-concurrent",
        "vision-envelope-over",
        "media-cache-thrash",
    } <= MEDIA_CASES
    assert "cold-long-64k" not in MEDIA_CASES
