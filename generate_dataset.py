import os
import glob
import random
import soundfile as sf
import torch
import torchaudio.transforms as T
import pandas as pd

TARGET_SR = 16000  # Tactical standard sample rate

def load_resample(file_path):
    try:
        data, sr = sf.read(file_path, dtype="float32")
    except Exception:
        return None
        
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        
    return waveform

def build_dataset(clean_dir, noise_dir, output_dir, num_samples=2500):
    output_clean_dir = os.path.join(output_dir, "clean")
    output_noisy_dir = os.path.join(output_dir, "noisy")
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    print("Scanning folders for audio files...")
    clean_files = glob.glob(os.path.join(clean_dir, "**", "*.flac"), recursive=True)
    noise_files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)

    print(f"Found {len(clean_files)} clean utterances.")
    print(f"Found {len(noise_files)} combat noise samples.")

    if len(clean_files) == 0 or len(noise_files) == 0:
        raise FileNotFoundError("Could not find audio files. Verify folder paths.")

    print(f"Synthesizing {num_samples} noisy-clean pairs...")
    records = []

    for i in range(num_samples):
        clean_path = random.choice(clean_files)
        noise_path = random.choice(noise_files)

        clean = load_resample(clean_path)
        noise = load_resample(noise_path)

        if clean is None or noise is None:
            continue

        c_len = clean.shape[1]
        n_len = noise.shape[1]
        if n_len < c_len:
            repeat_factor = (c_len // n_len) + 1
            noise = noise.repeat(1, repeat_factor)[:, :c_len]
        else:
            offset = random.randint(0, n_len - c_len)
            noise = noise[:, offset : offset + c_len]

        # Tactical SNR distribution between -10 dB and +15 dB
        snr_db = round(random.uniform(-10.0, 15.0), 1)
        clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-9)
        noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-9)
        scaled_noise = noise * (clean_rms / noise_rms) * (10.0 ** (-snr_db / 20.0))

        mixture = clean + scaled_noise

        # Prevent clipping
        max_peak = torch.max(torch.abs(mixture))
        if max_peak > 0.98:
            scale = 0.98 / max_peak
            mixture = mixture * scale
            clean = clean * scale

        file_id = f"synth_{i:05d}"
        clean_out = os.path.join(output_clean_dir, f"{file_id}.wav")
        noisy_out = os.path.join(output_noisy_dir, f"{file_id}.wav")

        sf.write(clean_out, clean.squeeze().numpy(), TARGET_SR)
        sf.write(noisy_out, mixture.squeeze().numpy(), TARGET_SR)

        records.append({
            "noisy_path": os.path.abspath(noisy_out),
            "clean_path": os.path.abspath(clean_out),
            "snr_db": snr_db
        })

        if (i + 1) % 250 == 0:
            print(f"Generated [{i + 1}/{num_samples}] pairs...")

    manifest_path = os.path.join(output_dir, "manifest.csv")
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"\n[DONE] Manifest generated at: {os.path.abspath(manifest_path)}")

if __name__ == "__main__":
    CLEAN_DIR = r"C:\Users\jeswa\Downloads\datasets\dev-clean\LibriSpeech\dev-clean"
    NOISE_DIR = r"C:\Users\jeswa\Downloads\datasets\MAD_dataset\training"
    OUTPUT_DIR = r"./data/train_set"

    build_dataset(CLEAN_DIR, NOISE_DIR, OUTPUT_DIR, num_samples=2500)