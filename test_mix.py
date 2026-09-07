import os
import soundfile as sf
import torch
import torchaudio.transforms as T
import numpy as np

TARGET_SR = 16000  # Military standard sample rate[cite: 1]

def load_and_resample(file_path):
    data, sr = sf.read(file_path, dtype="float32")
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

def mix_test(clean_path, noise_path, target_snr_db=0.0):
    os.makedirs("test_out", exist_ok=True)
    
    clean = load_and_resample(clean_path)
    noise = load_and_resample(noise_path)

    # Tile or trim noise to match clean utterance length[cite: 3]
    c_len = clean.shape[1]
    n_len = noise.shape[1]
    if n_len < c_len:
        repeat_factor = (c_len // n_len) + 1
        noise = noise.repeat(1, repeat_factor)[:, :c_len]
    else:
        noise = noise[:, :c_len]

    # Calculate RMS energy for 0 dB SNR scaling[cite: 3]
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-9)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-9)
    
    snr_factor = 10.0 ** (-target_snr_db / 20.0)
    scaled_noise = noise * (clean_rms / noise_rms) * snr_factor

    mixture = clean + scaled_noise
    max_peak = torch.max(torch.abs(mixture))
    if max_peak > 0.98:
        mixture = mixture / max_peak * 0.98

    sf.write("test_out/clean_ref.wav", clean.squeeze().numpy(), TARGET_SR)
    sf.write("test_out/mixed_noisy.wav", mixture.squeeze().numpy(), TARGET_SR)
    
    print(f"\n[SUCCESS] Audio mixed at {target_snr_db} dB SNR!")
    print(f"Clean reference: {os.path.abspath('test_out/clean_ref.wav')}")
    print(f"Mixed noisy:     {os.path.abspath('test_out/mixed_noisy.wav')}")

if __name__ == "__main__":
    CLEAN_SAMPLE = r"C:\Users\jeswa\Downloads\datasets\test\1272-135031-0003.flac"
    NOISE_SAMPLE = r"C:\Users\jeswa\Downloads\datasets\test\0.wav"
    
    mix_test(CLEAN_SAMPLE, NOISE_SAMPLE, target_snr_db=0.0)