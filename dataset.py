import os
import torch
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset

class TacticalAudioDataset(Dataset):
    def __init__(self, manifest_csv, segment_len_sec=3.0, fs=16000):
        self.df = pd.read_csv(manifest_csv)
        self.chunk_samples = int(segment_len_sec * fs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        noisy, _ = sf.read(row["noisy_path"], dtype="float32")
        clean, _ = sf.read(row["clean_path"], dtype="float32")

        noisy_t = torch.from_numpy(noisy)
        clean_t = torch.from_numpy(clean)

        total_samples = noisy_t.shape[0]
        if total_samples > self.chunk_samples:
            start = torch.randint(0, total_samples - self.chunk_samples + 1, (1,)).item()
            noisy_chunk = noisy_t[start : start + self.chunk_samples]
            clean_chunk = clean_t[start : start + self.chunk_samples]
        else:
            pad = self.chunk_samples - total_samples
            noisy_chunk = torch.nn.functional.pad(noisy_t, (0, pad))
            clean_chunk = torch.nn.functional.pad(clean_t, (0, pad))

        return noisy_chunk, clean_chunk