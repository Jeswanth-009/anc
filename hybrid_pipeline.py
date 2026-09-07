import os
import torch
import soundfile as sf
import numpy as np
from gtcrn import GTCRN

TARGET_SR = 16000
N_FFT = 512
HOP_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VADGatedNLMS:
    """
    Normalized Least Mean Squares (NLMS) filter with Neural VAD Gating.
    Adapts exclusively during non-speech segments to cancel stationary engine drones
    without causing double-talk speech attenuation.
    """
    def __init__(self, order=64, mu=0.03, eps=1e-6):
        self.order = order
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(order, dtype=np.float32)
        self.ref_buf = np.zeros(order, dtype=np.float32)

    def filter_step(self, ai_sample, ref_sample, is_speech: bool):
        # Update delay-line buffer
        self.ref_buf[1:] = self.ref_buf[:-1]
        self.ref_buf[0] = ref_sample

        # Predict residual tone
        pred_tone = np.dot(self.weights, self.ref_buf)
        cleaned_sample = ai_sample - pred_tone

        # Tap adaptation: FREEZE if speech is present to preserve vocal harmonics
        if not is_speech:
            energy = np.dot(self.ref_buf, self.ref_buf) + self.eps
            self.weights += (self.mu / energy) * cleaned_sample * self.ref_buf

        return cleaned_sample

def run_hybrid_enhancement(noisy_path, clean_path=None, model_path="checkpoints/gtcrn_latest.pth"):
    device = torch.device(DEVICE)
    noisy_wav, sr = sf.read(noisy_path, dtype="float32")
    if noisy_wav.ndim > 1:
        noisy_wav = np.mean(noisy_wav, axis=1)

    model = GTCRN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    tensor_in = torch.from_numpy(noisy_wav).unsqueeze(0).to(device)
    window = torch.hann_window(N_FFT).to(device)

    # 1. Neural Stage (GTCRN Non-Stationary Denoising)
    with torch.inference_mode():
        noisy_stft = torch.stft(tensor_in, n_fft=N_FFT, hop_length=HOP_LEN, window=window, return_complex=True)
        r_out, i_out = model(noisy_stft.real, noisy_stft.imag)
        
        # Calculate Neural VAD from speech-band mask energy (Bins 10 to 110: ~300 Hz - 3.4 kHz)
        mask_mag = torch.sqrt(r_out[:, 10:110, :]**2 + i_out[:, 10:110, :]**2)
        vad_per_frame = (torch.mean(mask_mag, dim=1) > 0.35).squeeze().cpu().numpy()

        est_stft = torch.complex(r_out, i_out)
        enhanced = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LEN, window=window, length=len(noisy_wav))

    ai_cleaned = enhanced.squeeze().cpu().numpy()
    
    # 2. Classical Residual Stage (VAD-Gated NLMS)
    # Reference = residual stationary noise stripped by GTCRN
    ref_noise = noisy_wav - ai_cleaned
    nlms = VADGatedNLMS(order=64, mu=0.04)
    
    final_output = np.zeros_like(ai_cleaned)
    total_frames = len(vad_per_frame)

    for n in range(len(ai_cleaned)):
        frame_idx = min(n // HOP_LEN, total_frames - 1)
        is_speech = vad_per_frame[frame_idx]
        final_output[n] = nlms.filter_step(ai_cleaned[n], ref_noise[n], is_speech)

    final_output = np.clip(final_output, -0.98, 0.98)
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/hybrid_enhanced.wav"
    sf.write(out_path, final_output, TARGET_SR)
    print(f"[PIPELINE PASS] Saved hybrid output to: {out_path}")
    return final_output

if __name__ == "__main__":
    import pandas as pd
    manifest = pd.read_csv("./data/train_set/manifest.csv")
    sample = manifest.iloc[0]
    run_hybrid_enhancement(sample["noisy_path"], sample["clean_path"])