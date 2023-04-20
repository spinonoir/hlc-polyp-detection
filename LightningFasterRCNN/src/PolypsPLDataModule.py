from pyparsing import Optional
import pytorch_lightning as pl
import src.PolypsDataset as polyps_data
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypsPLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './data', num_classes: int = 2, num_workers: int = 4, pin_memory: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = num_classes

        self.augmentation = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
        pass
