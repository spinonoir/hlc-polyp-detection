from pyparsing import Optional
import pytorch_lightning as pl
import src.MaskedPolypDataset as mpd
# impoer torch Transformations
from torchvision import transforms

class MaskedPolypDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './data',num_classes: int = 2, num_workers: int = 4, pin_memory: bool = False):
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


    def prepare_data(self):
        pass

    def setup(self, stage = None):
        pass
