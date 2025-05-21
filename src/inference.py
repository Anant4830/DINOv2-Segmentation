from model import DinoSegModel
from transformers import AutoImageProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DinoSegModel().to(device)
model.load_state_dict(torch.load("/home/iiitb/Desktop/anant/playground/ProjectBytes/src/best_model.pth"))
model.eval()

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
img = Image.open("/home/iiitb/Desktop/anant/playground/ProjectBytes/airplane1.jpg").convert("RGB")
inputs = image_processor(img, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(inputs.pixel_values)
    pred = output.argmax(1).squeeze().cpu().numpy()

# plt.imshow(pred, cmap='nipy_spectral')
# plt.title("Prediction")
# plt.show()

#plt.imsave(os.path.join(output_dir, "prediction.png"), prediction, cmap="nipy_spectral")
plt.imsave("pred_airplane1.png", pred, cmap="nipy_spectral")