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
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[2:]
        img_pool = self.global_avg_pool(x)
        img_pool = F.interpolate(img_pool, size=size, mode='bilinear', align_corners=False)

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)

        x = torch.cat([x1, x2, x3, x4, img_pool], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return self.dropout(x)

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels=1024, num_classes=21):
        super().__init__()
        self.aspp = ASPP(in_channels, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)  # Upsample to original image size
        #x = F.interpolate(x, scale_factor=(224, 224), mode='bilinear', align_corners=False)  # Upsample to original image size
        return x

class DinoDeepLabV3SegModel(nn.Module):
    def __init__(self, freeze_dino=True, num_classes=21):
        super().__init__()
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-large")
        self.decoder = DeepLabV3PlusDecoder(in_channels=1024, num_classes=num_classes)

        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):  # [B, 3, 224, 224]
        B, _, H, W = pixel_values.shape  # get input size
        feats = self.dino(pixel_values).last_hidden_state  # [B, 257, 1024]
        feats = feats[:, 1:, :]  # remove CLS token
        feats = feats.reshape(-1, 16, 16, 1024).permute(0, 3, 1, 2)  # [B, 1024, 16, 16]
        seg_logits = self.decoder(feats)  # [B, num_classes, 224, 224]
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)  # match input size
        return seg_logits    

# Point 4 in Bonus Points     
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


