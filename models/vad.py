import os
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


class VADBase(ABC):
    """Abstract base class for Voice Activity Detection."""

    @abstractmethod
    def extract_speech_segments(self, video_path: str):
        """Extract speech segments from a video file."""
        pass

    def extract_audio(
        self, video_path: str, wav_out: Optional[str] = None, sr: int = 16000
    ):
        """Extract mono 16kHz WAV from video using ffmpeg. Reuses file if it exists."""
        if wav_out is None:
            wav_out = os.path.splitext(video_path)[0] + ".16k.wav"
        if os.path.exists(wav_out):
            return wav_out
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "wav",
            wav_out,
        ]
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return wav_out


class SileroVAD(VADBase):
    """Voice Activity Detection using Silero-VAD. <br>
    https://github.com/snakers4/silero-vad/tree/master
    """

    def __init__(self, sr: str = 16000):
        self.model = load_silero_vad()
        self.sr = sr

    def extract_speech_segments(self, video_path: str):
        # Extract audio from video
        wav = self.extract_audio(video_path, sr=self.sr)
        # Load audio and resample if needed
        audio = read_audio(wav, sampling_rate=self.sr)
        # Run VAD
        speech_timestamps = get_speech_timestamps(
            audio, self.model, return_seconds=True
        )
        return speech_timestamps


class NemoVAD(VADBase):
    """Voice Activity Detection using NVIDIA NeMo. <br>
    https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html"""

    def __init__(self, sr: int = 16000, prob_thr: float = 0.5):
        self.sr = sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prob_thr = prob_thr
        self.model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            "vad_marblenet"
        )
        self.model.to(self.device).eval()

    def extract_speech_segments(self, video_path: str):
        """
        Extract speech segments from a video file using NeMo VAD.

        Steps:
        1. Extract mono 16kHz WAV audio from the video.
        2. Load and resample audio if needed.
        3. Slide a window over the audio and run VAD model on each segment.
        4. Collect speech probabilities for each segment.
        5. Merge consecutive segments above the probability threshold into speech intervals.
        6. Return list of speech segments (start/end times in seconds).
        """
        # Extract audio from video
        wav = self.extract_audio(video_path, sr=self.sr)

        # Load audio and resample if needed
        y, file_sr = sf.read(wav)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if file_sr != self.sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=self.sr)

        win_s, hop_s = 0.5, 0.25  # Window and hop size in seconds
        win, hop = int(self.sr * win_s), int(self.sr * hop_s)
        probs, spans = [], []

        # Slide window and run VAD
        for start in range(0, max(len(y) - win, 0) + 1, hop):
            seg = y[start : start + win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            seg_tensor = (
                torch.tensor(seg, dtype=torch.float32).to(self.device).unsqueeze(0)
            )
            lengths = torch.tensor([len(seg)], dtype=torch.int64, device=self.device)
            with torch.no_grad():
                logits = self.model.forward(
                    input_signal=seg_tensor, input_signal_length=lengths
                )
                # Get speech probability for segment
                if logits.ndim == 2 and logits.shape[-1] == 2:
                    p_speech = torch.softmax(logits, dim=-1)[0, 1].item()
                else:
                    p_speech = torch.sigmoid(logits.squeeze()).item()
            probs.append(p_speech)
            spans.append((start / self.sr, (start + win) / self.sr))

        # Merge segments above threshold
        segs = []
        cur = None
        for (s, e), p in zip(spans, probs):
            if p >= self.prob_thr:
                if cur is None:
                    cur = [s, e]
                elif s <= cur[1] + 1e-3:
                    cur[1] = e
                else:
                    segs.append((cur[0], cur[1]))
                    cur = [s, e]
            else:
                if cur is not None:
                    segs.append((cur[0], cur[1]))
                    cur = None
        if cur is not None:
            segs.append((cur[0], cur[1]))

        # Return speech segments
        return [{"start": round(s, 3), "end": round(e, 3)} for s, e in segs]
