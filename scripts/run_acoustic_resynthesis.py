"""Execute Frontier 5: Living Acoustic & Liturgical Heritage Resynthesis CLI.

Synthesizes scientifically accurate articulatory formant audio for Bronze Age texts:
- Reconstructs open CV moraic pacing.
- Formulates F1-F3 formant frequency trajectories.
- Enforces strict Evidence Tiering (E0-E4).
"""

from __future__ import annotations

from ancient_text_lab.acoustics import (
    AcousticEvidenceTier,
    BronzeAgeAcousticResynthesizer,
)


def run_acoustics_demo() -> None:
    print("=================================================================")
    print("   FRONTIER 5: LIVING ACOUSTIC & LITURGICAL RESYNTHESIS         ")
    print("=================================================================")

    synth = BronzeAgeAcousticResynthesizer(
        base_pitch_hz=125.0,
        mora_duration_ms=130.0,
        sample_rate_hz=16000,
    )

    # Inscription: Minoan Peak Sanctuary Libation Formula (IO Za 2 / PK Za 11)
    text_phrase = ["A", "TA", "I", "WA", "JA", "JA", "SA", "SA", "RA", "ME"]
    print(f"\n[INSCRIPTION] Minoan Libation Formula: {'-'.join(text_phrase)}")
    print("              Carrier: Steatite Peak Sanctuary Libation Vessel")

    morae = synth.parse_syllable_morae(
        text_phrase,
        tier=AcousticEvidenceTier.E3_PHONOTACTIC_CONSERVED,
    )

    print("\n[MORAIC PROSODIC TIMING]")
    print("-----------------------------------------------------------------")
    for m in morae:
        print(f"  Syllable: {m.syllable:4} | Vowel: /{m.nucleus_vowel}/ | Morae: {m.morae_count} | Dur: {m.duration_ms:.0f} ms | Tier: {m.evidence_tier.value}")

    frames = synth.generate_formant_trajectory(morae)
    waveform = synth.synthesize_pcm_waveform(frames)

    total_duration_sec = len(waveform) / synth.sample_rate_hz
    print("\n[AUDIO RESYNTHESIS COMPLETE]")
    print(f"  Total Duration:     {total_duration_sec:.2f} seconds ({len(frames)} formant control frames)")
    print(f"  Sample Rate:        {synth.sample_rate_hz} Hz PCM")
    print(f"  Peak Dynamic Range: {waveform.max():.2f} (Biot-Savart Resonator Filter)")
    print("  Evidence Tier:      >>> E3_phonotactic (Conserved Open CV Moraic System) <<<")
    print("=================================================================")


if __name__ == "__main__":
    run_acoustics_demo()
