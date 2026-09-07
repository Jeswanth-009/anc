import torch
import torch.nn as nn

class GTCRN(nn.Module):
    def __init__(self, n_fft=512, hop=128, n_erb=64, n_groups=4, hidden=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.freq_bins = n_fft // 2 + 1  # 257 bins at 16 kHz

        # Learnable subband compression matrix (257 bins -> 64 ERB bands)
        self.erb_W = nn.Parameter(torch.randn(n_erb, self.freq_bins) * 0.01)

        # Grouped 1D Convolutional Encoder (4 groups divide operations by 4x)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_erb * 2, hidden * n_groups, kernel_size=3, padding=1, groups=n_groups),
            nn.GroupNorm(n_groups, hidden * n_groups),
            nn.PReLU(),
            nn.Conv1d(hidden * n_groups, hidden * n_groups, kernel_size=3, padding=1, groups=n_groups),
            nn.GroupNorm(n_groups, hidden * n_groups),
            nn.PReLU()
        )

        # Dual-Path Grouped Recurrent Core
        self.intra_rnn = nn.GRU(hidden * n_groups, hidden, batch_first=True)
        self.inter_rnn = nn.GRU(hidden, hidden, batch_first=True)

        # Temporal Recurrent Attention (TRA) to suppress ballistic shockwave saturation
        self.tra_q = nn.Linear(hidden, hidden)
        self.tra_k = nn.Linear(hidden, hidden)
        self.tra_v = nn.Linear(hidden, hidden)

        # Complex Ratio Mask Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.PReLU(),
            nn.Linear(hidden * 2, self.freq_bins * 2)
        )

    def forward(self, real, imag):
        # real, imag shape: [Batch, Freq_Bins (257), Time_Frames]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        
        # Auditory subband compression across frequency
        erb = torch.einsum("ef, bft -> bet", self.erb_W, mag)
        feat = torch.cat([erb, erb], dim=1)  # [B, 128, T]

        # Grouped Convolutions
        x = self.encoder(feat)               # [B, 256, T]
        x = x.permute(0, 2, 1)               # [B, T, 256]

        # Dual-Path GRU Processing
        x, _ = self.intra_rnn(x)             # [B, T, 64]
        x, _ = self.inter_rnn(x)             # [B, T, 64]

        # Zero-lookahead attention
        B, T, C = x.shape
        q, k, v = self.tra_q(x), self.tra_k(x), self.tra_v(x)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5), dim=-1)
        x = x + torch.bmm(attn, v)

        # Full-range Complex Mask Prediction (tanh scaled by 2.0 enables phase cancellation)
        mask = (torch.tanh(self.decoder(x)) * 2.0).permute(0, 2, 1)  # [B, 514, T]
        Mr = mask[:, :self.freq_bins, :]
        Mi = mask[:, self.freq_bins:, :]

        # Complex-domain spectral multiplication
        out_real = real * Mr - imag * Mi
        out_imag = real * Mi + imag * Mr

        return out_real, out_imag