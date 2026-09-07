import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TacticalAudioDataset
from gtcrn import GTCRN

class CompositeTacticalLoss(nn.Module):
    def __init__(self, alpha=0.5, lt=0.2, ls=0.8):
        super().__init__()
        self.alpha, self.lt, self.ls, self.eps = alpha, lt, ls, 1e-8

    def forward(self, est_wav, tgt_wav, est_r, est_i, tgt_r, tgt_i):
        est_wav, tgt_wav = est_wav.float(), tgt_wav.float()
        est_r, est_i = est_r.float(), est_i.float()
        tgt_r, tgt_i = tgt_r.float(), tgt_i.float()

        # 1. Scale-Invariant SDR
        dot = torch.sum(est_wav * tgt_wav, dim=-1, keepdim=True)
        s_tgt = (dot / (torch.sum(tgt_wav**2, dim=-1, keepdim=True) + self.eps)) * tgt_wav
        e_noise = est_wav - s_tgt

        s_pwr = torch.sum(s_tgt**2, dim=-1) + self.eps
        n_pwr = torch.sum(e_noise**2, dim=-1) + self.eps
        ratio = torch.clamp(s_pwr / n_pwr, min=1e-6, max=1e6)
        si_sdr = 10.0 * torch.log10(ratio)
        loss_time = -torch.mean(si_sdr)

        # 2. Power-Compressed Complex Spectral Loss (alpha = 0.5)
        mag_est = torch.sqrt(est_r**2 + est_i**2 + self.eps)
        mag_tgt = torch.sqrt(tgt_r**2 + tgt_i**2 + self.eps)

        m_est = mag_est ** self.alpha
        m_tgt = mag_tgt ** self.alpha
        loss_mag = torch.mean((m_est - m_tgt) ** 2)

        scale_est = m_est / (mag_est + self.eps)
        scale_tgt = m_tgt / (mag_tgt + self.eps)

        r_est = est_r * scale_est
        i_est = est_i * scale_est
        r_tgt = tgt_r * scale_tgt
        i_tgt = tgt_i * scale_tgt
        loss_ri = torch.mean((r_est - r_tgt) ** 2 + (i_est - i_tgt) ** 2)

        return self.lt * loss_time + self.ls * (loss_mag + loss_ri)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device} ({torch.cuda.get_device_name(0)})")

    manifest_path = "./data/train_set/manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}. Run generate_dataset.py first.")

    dataset = TacticalAudioDataset(manifest_path, segment_len_sec=3.0)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)

    model = GTCRN().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded GTCRN Model: {param_count:,} trainable parameters.")

    criterion = CompositeTacticalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    n_fft, hop = 512, 128
    window = torch.hann_window(n_fft).to(device)
    epochs = 30
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\nStarting training across {epochs} epochs on 2,500 synthetic samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            noisy_stft = torch.stft(noisy, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            clean_stft = torch.stft(clean, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                r_out, i_out = model(noisy_stft.real, noisy_stft.imag)

            est_stft = torch.complex(r_out.float(), i_out.float())
            est_wav = torch.istft(est_stft, n_fft=n_fft, hop_length=hop, window=window, length=clean.shape[1])
            loss = criterion(est_wav, clean, r_out, i_out, clean_stft.real, clean_stft.imag)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Tactical Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "checkpoints/gtcrn_latest.pth")

    print("\n[SUCCESS] Model training complete.")
    print("Valid checkpoint saved to: checkpoints/gtcrn_latest.pth")

if __name__ == "__main__":
    train()