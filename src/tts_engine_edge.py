"""Microsoft Edge neural TTS wrapper with the same interface as TTSEngine."""
import asyncio
import os
import tempfile

import numpy as np
from pydub import AudioSegment


def _speed_to_rate(speed: float) -> str:
    """Convert speed multiplier (e.g. 1.2) to Edge-TTS rate string (e.g. '+20%')."""
    pct = round((speed - 1.0) * 100)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


class TTSEngineEdge:
    """
    Microsoft Edge neural TTS (online).

    Uses the same synthesize() / unload() interface as TTSEngine so it can
    be swapped into the pipeline without any other changes.

    Vietnamese voices:
        vi-VN-HoaiMyNeural   — female (default)
        vi-VN-NamMinhNeural  — male
    """

    def __init__(
        self,
        voice: str = "vi-VN-HoaiMyNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        sample_rate: int = 24000,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Convert *text* to audio via Edge-TTS.

        Returns:
            (wav, sample_rate) where wav is a 1-D float32 numpy array.
        """
        wav = asyncio.run(self._synthesize_async(text))
        return wav, self.sample_rate

    async def _synthesize_async(self, text: str) -> np.ndarray:
        import edge_tts  # type: ignore  (installed separately)

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            await communicate.save(tmp_path)
            audio = AudioSegment.from_mp3(tmp_path)
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            wav = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        finally:
            os.unlink(tmp_path)

        return wav

    def unload(self) -> None:
        """No-op: Edge-TTS has no local model to unload."""
        pass
