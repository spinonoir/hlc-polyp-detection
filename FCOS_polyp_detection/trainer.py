import os
from pathlib import Path
import torch
from src.LightningModuleFCOS import LightningModuleFCOS as pl_module

CONFIG = dict (
    project = "hlc-polyp-detection",
    architecture = "fcos_resnet50_fpn",
    dataset_id = "hlc-custom-polyp-detection",
    infra = "osx",
    num_classes = 2,
    max_epochs = 100,
    lr=0.01,
    min_lr=0.0000001,
    epochs=15,
    batch_size=4,
    nesterov=True,
    momentum=0.9,
    weight_decay=0.0005,
    clip_limit=0.25,
    difference=False,
    name="fancy_walrus"
)

ROOT_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models/fcos_resnet50_fpn")
LOG_DIR = os.path.join(ROOT_DIR, "log")

NUM_CLASSES = 2

BATCH_SIZE = 4
INPUT_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4



from src.PolypsPLDataModule import PolypsPLDataModule

from src.LightningModuleFCOS import LightningModuleFCOS as pl_module
from pytorch_lightning import Trainer

data_dir = 'data'

if __name__ == "__main__":
    print(os.getcwd())
    polyp_dm = PolypsPLDataModule(data_dir=data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = pl_module(CONFIG)
    trainer = Trainer(
        log_every_n_steps=1,
        max_epochs=1
    )
    trainer.fit(model, polyp_dm)