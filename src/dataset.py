import torch

from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from PIL import Image

#train_dataset = VOCSegmentation(root='data/', year='2012', image_set='train', download=True)

# writing custom dataset, inheriting from VOCSegmentation dataset
class VOCSegmentationWithPIL(VOCSegmentation):
    def __init__(self, root='data', year='2012', image_set='train',
                 download=True, image_size=(224, 224)):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.image_resize = T.Resize(image_size)
        self.mask_transform = T.Compose([
            T.Resize(image_size, interpolation=Image.NEAREST),
            T.PILToTensor(),  # Keeps label values intact
        ])

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        image = self.image_resize(image)  # still PIL.Image
        mask = self.mask_transform(mask).squeeze(0).long()  # [H, W] as LongTensor
        return image, mask
    

def collate_fn_pil(batch):
    images, masks = zip(*batch)  # tuple of lists
    return list(images), torch.stack(masks)  # keep images as list of PIL