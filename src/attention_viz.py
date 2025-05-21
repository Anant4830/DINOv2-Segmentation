import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoImageProcessor
from model import DinoSegModelWithAttention

# ========== CONFIG ========== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "/home/iiitb/Desktop/anant/playground/ProjectBytes/airplane1.jpg"  
output_dir = "inference_attention_outputs"
os.makedirs(output_dir, exist_ok=True)

# ========== Load Model ========== #
model = DinoSegModelWithAttention().to(device)
model.load_state_dict(torch.load("/home/iiitb/Desktop/anant/playground/ProjectBytes/src/best_model.pth", map_location=device))
model.eval()

# ========== Load Image ========== #
image = Image.open(image_path).convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
inputs = image_processor(image, return_tensors="pt").to(device)

# ========== Inference + Attention ========== #
with torch.no_grad():
    output, attentions = model(inputs.pixel_values, return_attention=True)
    prediction = output.argmax(1).squeeze().cpu().numpy()

# ========== Process Attention Map ========== #
attn = attentions[-1][0, 0]  # last layer, head 0 → shape [num_tokens, num_tokens]
cls_attn = attn[0, 1:]       # CLS token → all patches (exclude self)

# Reshape to (H, W)
num_patches = cls_attn.shape[0]
side = int(num_patches**0.5)
attn_map = cls_attn.reshape(side, side).cpu().numpy()

# Upsample to image size
attn_resized = zoom(attn_map, (image.height / side, image.width / side))

# ========== Save All Results ========== #
# 1. Original Image
image.save(os.path.join(output_dir, "input_image.jpg"))

# 2. Segmentation Prediction
plt.imsave(os.path.join(output_dir, "prediction.png"), prediction, cmap="nipy_spectral")

# 3. Attention Map
plt.imsave(os.path.join(output_dir, "attention_map.png"), attn_resized, cmap="inferno")

# 4. Combined Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(attn_resized, cmap="inferno")
plt.title("DINOv2 Attention Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap="nipy_spectral")
plt.title("Segmentation Prediction")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_output.png"))
plt.close()
