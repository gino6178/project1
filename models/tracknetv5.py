import torch
import torch.nn as nn
import torch.nn.functional as F

class R2Plus1DBlock(nn.Module):
    """
    R(2+1)D Spatiotemporal Block
    將 3D 卷積分解為 2D 空間卷積與 1D 時間卷積
    """
    def __init__(self, in_channels, out_channels, k_s=3, k_t=3):
        super(R2Plus1DBlock, self).__init__()
        # Spatial Conv: 1 x 3 x 3
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, k_s, k_s),
                                      padding=(0, k_s // 2, k_s // 2), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.SiLU(inplace=True) # 使用 SiLU 保持一致性
        # Temporal Conv: k_t x 1 x 1
        self.temporal_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(k_t, 1, 1),
                                       padding=(k_t // 2, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        h = self.relu(self.bn1(self.spatial_conv(x)))
        h = self.relu(self.bn2(self.temporal_conv(h)))
        return h

class GatedResidualMotionPrompt(nn.Module):
    """
    Gated Residual Motion Prompt Layer
    使用門控機制同時抑制雜訊並保留動態細節
    """
    def __init__(self, channels=1):
        super(GatedResidualMotionPrompt, self).__init__()
        # 溫度參數與線性轉換參數
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # Gate Path (g)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_G = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Transformation Path (T)
        self.conv_T = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 轉換為灰階 (假設輸入為 RGB 或 RGB+Diff，取平均作為強度)
        gray = x.mean(dim=2, keepdim=True) # (B, T, 1, H, W)

        # 計算相鄰影格絕對差值並取時間平均 (D)
        diffs = [torch.abs(gray[:, i+1] - gray[:, i]) for i in range(T - 1)]
        D = torch.stack(diffs, dim=1).mean(dim=1) # (B, 1, H, W)

        # Gate Path 計算
        tau_clamped = torch.clamp(self.tau, min=0.25, max=4.0)
        g_feat = self.conv_G(self.avg_pool(D))
        g = torch.sigmoid(g_feat / tau_clamped)

        # Transformation Path 計算
        t_feat = self.conv_T(D)

        # 殘差融合與最終 Motion Map 生成
        F_map = (1 - g) * D + g * t_feat
        M = torch.sigmoid(self.alpha * F_map + self.beta)

        return M

class SCG(nn.Module):
    """
    Spatial-Channel-Gate (SCG) Module
    在解碼器中融合視覺特徵與動態先驗
    """
    def __init__(self, v_channels, m_channels=1):
        super(SCG, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Channel Gate
        self.mlp = nn.Sequential(
            nn.Linear(v_channels + m_channels, (v_channels + m_channels) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((v_channels + m_channels) // 2, v_channels)
        )

        # Spatial Gate
        self.spatial_conv = nn.Conv2d(m_channels, 1, kernel_size=3, padding=1)

        # Mixing Block N (Depthwise-separable)
        self.mix_depthwise = nn.Conv2d(v_channels, v_channels, kernel_size=3, padding=1, groups=v_channels)
        self.mix_pointwise = nn.Conv2d(v_channels, v_channels, kernel_size=1)

    def forward(self, V, M):
        B, C, H, W = V.shape

        # Channel Gate (gc)
        v_gap = self.gap(V).view(B, -1)
        m_gap = self.gap(M).view(B, -1)
        concat_gap = torch.cat([v_gap, m_gap], dim=1)
        gc = torch.sigmoid(self.mlp(concat_gap)).view(B, C, 1, 1)

        # Spatial Gate (gs)
        gs = torch.sigmoid(self.spatial_conv(V.mean(dim=1, keepdim=True)))

        # Gated Modulation
        gated_V = (V * gs) * gc

        # Mixing & Residual
        mix_out = self.mix_pointwise(self.mix_depthwise(gated_V))
        return V + mix_out

class FusedMBConvBlock(nn.Module):
    """
    Fused-MBConv for Encoder
    使用標準 3x3 卷積取代 depthwise，加速初期特徵提取
    """
    def __init__(self, in_dim, out_dim, expand_ratio=4):
        super(FusedMBConvBlock, self).__init__()
        hidden_dim = in_dim * expand_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.use_res = (in_dim == out_dim)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.project(x)
        if self.use_res:
            x = x + res
        return x

class MBConvBlock(nn.Module):
    """
    Standard MBConv for Decoder
    """
    def __init__(self, in_dim, out_dim, expand_ratio=4):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_dim * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.use_res = (in_dim == out_dim)

    def forward(self, x):
        res = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_res:
            x = x + res
        return x

class TrackNetV5(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super(TrackNetV5, self).__init__()
        self.alpha = alpha
        self.out_dim = out_dim # T
        self.in_channels = in_dim // out_dim # 3 (RGB) or 4 (Subtract+Concat)

        def scale(ch):
            return max(1, int(ch * alpha))

        # 1. 動態先驗提取
        self.motion_prompt = GatedResidualMotionPrompt(channels=1)

        # 2. R(2+1)D 區塊
        self.r2plus1d = R2Plus1DBlock(self.in_channels, self.in_channels)

        # Backbone 初始通道數：R(2+1)D 輸出通道 * 時間影格數
        backbone_in_ch = self.in_channels * self.out_dim

        # 3. Encoder (Fused-MBConv)
        self.down_block_1 = nn.Sequential(
            nn.Conv2d(backbone_in_ch, scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            FusedMBConvBlock(scale(128), scale(128))
        )
        self.down_block_2 = nn.Sequential(
            nn.Conv2d(scale(128), scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            FusedMBConvBlock(scale(128), scale(128))
        )
        self.down_block_3 = nn.Sequential(
            nn.Conv2d(scale(128), scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            FusedMBConvBlock(scale(128), scale(128)),
            FusedMBConvBlock(scale(128), scale(128))
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(scale(128), scale(256), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(256)),
            nn.SiLU(inplace=True),
            FusedMBConvBlock(scale(256), scale(256)),
            FusedMBConvBlock(scale(256), scale(256))
        )

        # 4. Decoder (MBConv + SCG)
        self.up_block_1 = nn.Sequential(
            nn.Conv2d(scale(256) + scale(128), scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            MBConvBlock(scale(128), scale(128))
        )
        self.scg_1 = SCG(scale(128))

        self.up_block_2 = nn.Sequential(
            nn.Conv2d(scale(128) + scale(128), scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            MBConvBlock(scale(128), scale(128))
        )
        self.scg_2 = SCG(scale(128))

        self.up_block_3 = nn.Sequential(
            nn.Conv2d(scale(128) + scale(128), scale(128), 3, padding=1, bias=False),
            nn.BatchNorm2d(scale(128)),
            nn.SiLU(inplace=True),
            MBConvBlock(scale(128), scale(128))
        )
        self.scg_3 = SCG(scale(128))

        self.predictor = nn.Conv2d(scale(128), out_dim, (1, 1))

        # 深監督 (L_ds)：兩個較粗的解碼階段各接一個 1x1 頭，上採樣回全解析度。
        # 權重零初始化，第 0 步不會有梯度流回解碼器。
        self.aux1 = nn.Conv2d(scale(128), out_dim, (1, 1))
        self.aux2 = nn.Conv2d(scale(128), out_dim, (1, 1))
        for h in (self.aux1, self.aux2):
            nn.init.zeros_(h.weight)
            nn.init.constant_(h.bias, -4.595)

    def forward(self, x):
        B, C_total, H, W = x.shape
        T = self.out_dim
        C = self.in_channels
        x = x.view(B, T, C, H, W)

        # === 動態分支 (Motion Branch) ===
        M = self.motion_prompt(x)

        # 將 M 降採樣以符合 Decoder 各層空間解析度
        M3 = M                                              # 對應 H
        M2 = F.max_pool2d(M, 2, 2)                          # 對應 H/2
        M1 = F.max_pool2d(M, 4, 4)                          # 對應 H/4

        # === 視覺分支 (Visual Branch) ===
        # 轉為 (B, C, T, H, W) 傳入 R(2+1)D
        x_3d = x.permute(0, 2, 1, 3, 4)
        x_3d = self.r2plus1d(x_3d)

        # 展平 T 視為 Channel 傳入 2D Encoder: (B, T*C, H, W)
        x_2d = x_3d.permute(0, 2, 1, 3, 4).reshape(B, x_3d.shape[1] * T, H, W)

        # Encoder
        x1 = self.down_block_1(x_2d)
        p1 = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
        x2 = self.down_block_2(p1)
        p2 = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
        x3 = self.down_block_3(p2)
        p3 = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)

        b = self.bottleneck(p3)

        # Decoder + SCG
        u1 = torch.cat([nn.Upsample(scale_factor=2, mode='nearest')(b), x3], dim=1)
        d1 = self.scg_1(self.up_block_1(u1), M1)

        u2 = torch.cat([nn.Upsample(scale_factor=2, mode='nearest')(d1), x2], dim=1)
        d2 = self.scg_2(self.up_block_2(u2), M2)

        u3 = torch.cat([nn.Upsample(scale_factor=2, mode='nearest')(d2), x1], dim=1)
        d3 = self.scg_3(self.up_block_3(u3), M3)

        out = self.predictor(d3)
        if self.training:
            a1 = F.interpolate(self.aux1(d1), size=(H, W), mode='bilinear',
                               align_corners=False)
            a2 = F.interpolate(self.aux2(d2), size=(H, W), mode='bilinear',
                               align_corners=False)
            return torch.sigmoid(out), [torch.sigmoid(a1), torch.sigmoid(a2)]
        return torch.sigmoid(out)
