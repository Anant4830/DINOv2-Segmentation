import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        logits = torch.softmax(logits, dim=1)
        #one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # Clamp values outside valid class index range to ignore_index
        targets = torch.where(targets >= num_classes, torch.tensor(self.ignore_index).to(targets.device), targets)

        # Only create one-hot if within valid range
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0  # Temporarily set ignored positions to class 0 to avoid one-hot crash

        one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        
        # Ignore masked values
        mask = (targets != self.ignore_index).unsqueeze(1)
        logits = logits * mask
        one_hot = one_hot * mask

        intersection = (logits * one_hot).sum(dim=(0, 2, 3))
        union = logits.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceCELoss(torch.nn.Module):
    def __init__(self, ce_weight=0.5, ignore_index=255):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss

