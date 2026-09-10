"""Tests for Frontier 5: Living Acoustic & Liturgical Heritage Resynthesis."""

from __future__ import annotations

import numpy as np
from ancient_text_lab.acoustics import (
    AcousticEvidenceTier,
    BronzeAgeAcousticResynthesizer,
)


def test_bronze_age_acoustic_resynthesis() -> None:
    synth = BronzeAgeAcousticResynthesizer(
        base_pitch_hz=120.0, mora_duration_ms=120.0, sample_rate_hz=16000
    )

    # Reconstruct the Minoan libation formula phrase: JA-SA-SA-RA-ME
    syllables = ["ja", "sa", "sa", "ra", "me"]
    morae = synth.parse_syllable_morae(
        syllables, tier=AcousticEvidenceTier.E3_PHONOTACTIC_CONSERVED
    )

    assert len(morae) == 5
    assert all(m.morae_count == 1 for m in morae)
    assert morae[0].duration_ms == 120.0
    assert morae[0].evidence_tier == AcousticEvidenceTier.E3_PHONOTACTIC_CONSERVED

    # Generate formant trajectory frames
    frames = synth.generate_formant_trajectory(morae)
    assert len(frames) > 0
    # Total duration = 5 syllables * 120 ms = 600 ms -> ~60 frames at 10ms steps
    assert abs(frames[-1].time_ms - 590.0) < 20.0

    # Synthesize PCM audio waveform
    waveform = synth.synthesize_pcm_waveform(frames)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert len(waveform) > 0
    # Peak amplitude bounded and normalized
    assert np.max(np.abs(waveform)) <= 1.0
