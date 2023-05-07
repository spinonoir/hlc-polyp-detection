from pyparsing import Optional
import pytorch_lightning as pl
import src.PolypsDataset as polyps_data
# import DataLoader from PyTorch
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import DEVICE, BATCH_SIZE, NUM_WORKERS



def collate_fn(batch):
    return tuple(zip(*batch))

class PolypsPLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=BATCH_SIZE, num_classes=2, num_workers=NUM_WORKERS, pin_memory=True, resize_to=800):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.resize_to = resize_to


        self.bbox_params = A.BboxParams(
            format='pascal_voc', 
            min_visibility=0.1, 
            min_area=128, 
            label_fields=['labels'])

        self.train_transform = A.Compose([
                A.Flip(p=0.25),
                A.RandomRotate90(p=0.15),
                A.RandomBrightnessContrast(p=0.25),
                A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                ToTensorV2(p=1.0),],
            bbox_params=self.bbox_params)
        
        self.test_transform = A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params=self.bbox_params)

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = polyps_data.PolypsDataset(self.data_dir, stage='train', transforms=self.train_transform, resize_to=self.resize_to)
            self.val_dataset = polyps_data.PolypsDataset(self.data_dir, stage='validation', transforms=self.test_transform, resize_to=self.resize_to)
        if stage == 'test' or stage is None:
            self.test_dataset = polyps_data.PolypsDataset(self.data_dir, stage='test', transforms=self.test_transform, resize_to=self.resize_to)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

   