import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model

# for deepLab
from transformers import AutoModel
import segmentation_models_pytorch as smp

class UNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, num_classes=21):
        super().__init__()

        # Upsample from 16x16 → 32x32 → 64x64 → 128x128 → 224x224 (final)
        self.up1 = self._upsample_block(in_channels, 512)
        self.up2 = self._upsample_block(512, 256)
        self.up3 = self._upsample_block(256, 128)
        self.up4 = self._upsample_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)  # 16 → 32
        x = self.up2(x)  # 32 → 64
        x = self.up3(x)  # 64 → 128
        x = self.up4(x)  # 128 → ~256
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # match input size
        x = self.final_conv(x)
        return x  # [B, num_classes, 224, 224]
    
class DinoSegModel(nn.Module):
    def __init__(self, freeze_dino=True, num_classes=21):
        super().__init__()
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-large")
        self.decoder = UNetDecoder(in_channels=1024, num_classes=num_classes)

        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):  # [B, 3, 224, 224]
        feats = self.dino(pixel_values).last_hidden_state  # [B, 257, 1024]
        feats = feats[:, 1:, :]  # remove CLS token
        feats = feats.reshape(-1, 16, 16, 1024).permute(0, 3, 1, 2)  # [B, 1024, 16, 16]
        seg_logits = self.decoder(feats)  # [B, num_classes, 224, 224]
        return seg_logits

#  Point 5 in Bonus Points 
# to visualize the attention map at INFERENCE time
class DinoSegModelWithAttention(nn.Module):
    def __init__(self, num_classes=21, freeze_dino=True):
        super().__init__()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-large", output_attentions=True)
        self.hidden_dim = 1024

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.decoder = UNetDecoder(in_channels=self.hidden_dim, num_classes=num_classes)

    def forward(self, x, return_attention=False):
        outputs = self.dino(x)
        features = outputs.last_hidden_state  # [B, N, C]
        attentions = outputs.attentions       # List of attention tensors per layer

        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features[:, 1:, :]  # Remove CLS token
        features = features.permute(0, 2, 1).reshape(B, C, H, W)

        seg_output = self.decoder(features)

        if return_attention:
            return seg_output, attentions
        return seg_output


# Point 3 in Bonus Points    
# DeepLab model for comparison
class DinoDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, freeze_dino=True):
        super().__init__()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-large")
        self.freeze_dino = freeze_dino

        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # SMP requires an encoder with feature extraction
        # We mimic DINO features as an encoder output here.
        # You may need to map DINO's outputs accordingly.
        self.seg_head = smp.DeepLabV3Plus(
            encoder_name=None,             # We're passing our own encoder
            encoder_weights=None,
            in_channels=1024,              # DINOv2-large last hidden size
            classes=num_classes,
        )

    def forward(self, x):
        features = self.dino(x).last_hidden_state  # [B, num_patches, C]
        B, N, C = features.shape
        H = W = int(N ** 0.5)  # assuming square image
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # reshape for SMP
        return self.seg_head(features)

# Point 3 in Bonus Points     
# attention based model:
import torch.nn as nn
from transformers import AutoModel

class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, num_layers=2):
        super().__init__()
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W] → flatten
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        x = self.transformer_decoder(x)  # [B, N, C]
        x = self.cls_head(x)             # [B, N, num_classes]

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, num_classes, H, W]
        return x


class DinoAttentionSegModel(nn.Module):
    def __init__(self, num_classes=21, freeze_dino=True):
        super().__init__()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-large")
        self.hidden_dim = 1024  # DINOv2-Large output

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.decoder = AttentionDecoder(embed_dim=self.hidden_dim, num_classes=num_classes)

    def forward(self, x):
        # DINOv2 outputs [B, N, C], need to reshape into spatial map
        features = self.dino(x).last_hidden_state  # [B, N, C]
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]

        return self.decoder(features)


