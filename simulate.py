import os
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from gtcrn import GTCRN

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

TARGET_SR = 16000
N_FFT = 512
HOP = 128

def compute_sisdr(ref, est):
    """DRDO standard Scale-Invariant Signal-to-Distortion Ratio."""
    eps = 1e-8
    # Optimal scaling projection
    scale = np.dot(est, ref) / (np.sum(ref ** 2) + eps)
    target = scale * ref
    noise = est - target
    return 10.0 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))

def run_simulation(noisy_path, clean_path, model_path="checkpoints/gtcrn_latest.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Audio
    noisy_np, sr = sf.read(noisy_path, dtype="float32")
    clean_np, _ = sf.read(clean_path, dtype="float32")
    min_len = min(len(noisy_np), len(clean_np))
    noisy_np, clean_np = noisy_np[:min_len], clean_np[:min_len]

    # 2. Load Model
    model = GTCRN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Complex STFT
    noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    window = torch.hann_window(N_FFT).to(device)

    with torch.no_grad():
        noisy_stft = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)
        r_in, i_in = noisy_stft.real, noisy_stft.imag

        # Neural Denoising Forward Pass
        r_out, i_out = model(r_in, i_in)

        # Inverse STFT Synthesis
        est_stft = torch.complex(r_out, i_out)
        enhanced_wav = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP, window=window, length=min_len)

    ai_cleaned = np.clip(enhanced_wav.squeeze().cpu().numpy(), -0.98, 0.98)

    # 4. Save Output File
    os.makedirs("results", exist_ok=True)
    out_path = "results/enhanced_output.wav"
    sf.write(out_path, ai_cleaned, TARGET_SR)
    print(f"\n[SAVED] Enhanced audio written to: {os.path.abspath(out_path)}")

    # 5. Measure Metrics using SI-SDR
    sdr_before = compute_sisdr(clean_np, noisy_np)
    sdr_after = compute_sisdr(clean_np, ai_cleaned)
    sdr_gain = sdr_after - sdr_before

    print("\n" + "=" * 58)
    print("           TACTICAL PERFORMANCE REPORT           ")
    print("=" * 58)
    print(f"Metric       | Before Clean | After GTCRN  | Status")
    print("-" * 58)
    print(f"SI-SDR (dB)  |   {sdr_before:6.2f} dB |    {sdr_after:6.2f} dB | {'PASS' if sdr_gain >= 10.0 else 'ANALYSIS'} (+{sdr_gain:.2f} dB)")

    if HAS_STOI:
        stoi_before = stoi(clean_np, noisy_np, TARGET_SR)
        stoi_after = stoi(clean_np, ai_cleaned, TARGET_SR)
        print(f"STOI Score   |   {stoi_before:6.3f}    |    {stoi_after:6.3f}    | {'PASS (>0.85)' if stoi_after >= 0.85 else 'ANALYSIS'}")
    else:
        print("STOI Score   |   [pystoi not installed]")

    print("=" * 58)

if __name__ == "__main__":
    manifest = pd.read_csv("./data/train_set/manifest.csv")
    sample = manifest.iloc[0]

    print(f"Testing on Sample: {sample['noisy_path']}")
    print(f"Simulated Input SNR: {sample['snr_db']} dB")

    run_simulation(sample['noisy_path'], sample['clean_path'])