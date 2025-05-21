from dataset import VOCSegmentationWithPIL
from dataset import collate_fn_pil
from model import DinoSegModel
#from model import DinoAttentionSegModel
#from model import DinoDeepLabV3SegModel

from transformers import AutoImageProcessor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from torchmetrics.classification import MulticlassJaccardIndex

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os



#--------------loading train data data loader and validation dataloader---------------------
train_dataset = VOCSegmentationWithPIL(
    root='data_train',
    year='2012',
    image_set='train',
    download=True,
    image_size=(224, 224)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn_pil,
    pin_memory=True
)


val_dataset = VOCSegmentationWithPIL(
    root='data_val',
    year='2012',
    image_set='val',
    download=True,
    image_size=(224, 224)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn_pil,
    pin_memory=True
)


# ------------------ Setup ------------------
num_classes = 21
ignore_index = 255
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = DinoSegModel(freeze_dino=True).to(device)
#model = DinoDeepLabV3SegModel(freeze_dino=False).to(device)
#model = DinoAttentionSegModel(num_classes=21, freeze_dino=True).to(device)


image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

# Loss, Optimizer, Scheduler
#criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)


#lovasz_losses
# from lovasz_losses import lovasz_softmax
# def lovasz_loss_fn(logits, targets):
#     return lovasz_softmax(logits, targets, ignore=255)
# criterion = lovasz_loss_fn

#dice+CE loss
from diceCE import DiceCELoss
criterion = DiceCELoss(ignore_index=255)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Metric
miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(device)

# Training config
num_epochs = 20
train_losses, val_losses, val_ious = [], [], []

# Paths
best_model_path = "best_model.pth"
checkpoint_path = "last_checkpoint.pth"
best_val_loss = float('inf')
start_epoch = 0

# ------------------ Load from Checkpoint (if exists) ------------------
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed training from checkpoint at epoch {start_epoch}")

# ------------------ Validation ------------------
def evaluate(model, loader, criterion, image_processor, device, epoch):
    model.eval()
    val_loss = 0.0
    miou_metric.reset()
    save_dir = os.path.join("val_preds", f"epoch_{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        val_loop = tqdm(loader, desc=f"Validation Epoch {epoch+1}")
        for step, (images, masks) in enumerate(val_loop):
            inputs = image_processor(images, return_tensors='pt').to(device)
            masks = masks.to(device)
            outputs = model(inputs.pixel_values)

            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_loop.set_postfix(val_loss=loss.item())

            preds = outputs.argmax(dim=1)
            miou_metric.update(preds, masks)

            # Save every sample in the batch
            for i in range(len(images)):
                image_np = images[i]
                mask_np = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()

                #plt.imsave(os.path.join(save_dir, f"image_{step}_{i}.jpg"), image_np)
                plt.imsave(os.path.join(save_dir, f"gt_{step}_{i}.png"), mask_np, cmap='nipy_spectral')
                plt.imsave(os.path.join(save_dir, f"pred_{step}_{i}.png"), pred_np, cmap='nipy_spectral')

    return val_loss / len(loader), miou_metric.compute().item()

# ------------------ Training Loop ------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, masks in loop:
        inputs = image_processor(images, return_tensors='pt').to(device)
        masks = masks.to(device)

        outputs = model(inputs.pixel_values)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluate only every 10 epochs
    if (epoch + 1) % 20 == 0:
        avg_val_loss, val_miou = evaluate(model, val_loader, criterion, image_processor, device, epoch)
        scheduler.step(avg_val_loss)
        val_losses.append(avg_val_loss)
        val_ious.append(val_miou)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
    else:
        val_losses.append(None)
        val_ious.append(None)

    # Save checkpoint every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")


import matplotlib.pyplot as plt

plt.figure()

# Plot train loss (has value every epoch)
plt.plot(range(num_epochs), train_losses, label='Train Loss')

# Prepare filtered val loss and x-ticks
val_epochs = [i for i, v in enumerate(val_losses) if v is not None]
val_values = [v for v in val_losses if v is not None]

plt.plot(val_epochs, val_values, label='Validation Loss', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()