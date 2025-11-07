"""Audio synthesis utilities for eigenmodes."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf

from .solver import EigenResult, modes_to_frequencies


def synthesize_overtones(
    result: EigenResult,
    modes: Sequence[int] | None = None,
    *,
    duration: float = 2.0,
    sample_rate: int = 44100,
    wave_speed: float = 340.0,
    amplitudes: Sequence[float] | None = None,
    envelope: callable | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Create an audio waveform combining selected eigenmodes."""

    if modes is None:
        modes = list(range(min(6, len(result.values))))
    if not modes:
        raise ValueError("At least one mode must be selected for synthesis")

    freqs = modes_to_frequencies(result.values, wave_speed=wave_speed)
    times = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(times)

    if amplitudes is None:
        amplitudes = np.linspace(1.0, 0.3, num=len(modes))

    if envelope is None:
        envelope = lambda t: 0.5 * (1 - np.cos(2 * np.pi * t / duration))  # cosine fade

    mode_fields = [result.mode(idx) for idx in modes]
    weights = [np.linalg.norm(field[result.domain.mask]) for field in mode_fields]
    weights = np.array(weights)
    weights[weights == 0.0] = 1.0

    for amp, freq, weight in zip(amplitudes, freqs[modes], weights):
        phase = amp / weight * np.sin(2 * np.pi * freq * times)
        audio += phase

    env = envelope(times)
    audio = audio * env

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = 0.95 * audio / peak

    return audio.astype(np.float32)


def export_wav(audio: np.ndarray, file_path: str | Path, sample_rate: int = 44100) -> Path:
    file_path = Path(file_path)
    sf.write(file_path, audio, sample_rate)
    return file_path


def synthesize_and_save(
    result: EigenResult,
    file_path: str | Path,
    *,
    modes: Iterable[int] | None = None,
    duration: float = 2.0,
    sample_rate: int = 44100,
    wave_speed: float = 340.0,
) -> Path:
    audio = synthesize_overtones(
        result,
        modes=list(modes) if modes is not None else None,
        duration=duration,
        sample_rate=sample_rate,
        wave_speed=wave_speed,
    )
    return export_wav(audio, file_path, sample_rate=sample_rate)
