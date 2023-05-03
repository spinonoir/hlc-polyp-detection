import os
from pathlib import Path
import torch
import wandb
from contextlib import contextmanager
from src.LightningFasterModule import LightningFasterModule
from src.PolypsPLDataModule import PolypsPLDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


CONFIG = dict (
    project = "hlc-polyp-detection",
    architecture = "fasterrcnn_resnext50_32x4d",
    dataset_id = "hlc-custom-polyp-detection",
    infra = "osx",
    num_classes = 2,
    lr=0.01,
    min_lr=0.0000001,
    epochs=15,
    max_epochs=100,
    batch_size=4,
    nesterov=True,
    momentum=0.9,
    weight_decay=0.0005,
    clip_limit=0.25,
    difference=False,
)

ROOT_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
LOG_DIR = os.path.join(ROOT_DIR, "log")

NUM_CLASSES = 2

BATCH_SIZE = 4
INPUT_SIZE = 800
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

wandb_key = Path(os.path.join(ROOT_DIR, "wandb.txt")).read_text().strip()
os.environ["WANDB_API_KEY"] = wandb_key
os.environ["WANDB_NOTEBOOK_NAME"] = "initial_testing"

@contextmanager
def wandb_context(configuration=CONFIG):
    run = wandb.init(reinit=True, config=configuration, project=CONFIG['project'])
    try:
        yield run
    finally:
        wandb.finish()

def train_model(model, run_name, dm=None, run=None):
    wandb_logger = None
    if run is not None:
        run.config["train_run_name"] = run_name

        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger()
    
    chkpt = ModelCheckpoint(
        dirpath=os.path.join(MODEL_DIR, "chkpts"),
        filename=f"chkpt-{run_name}",
        monitor="val_recall",
        mode="max")
    
    trainer = Trainer(gpus=1, logger=wandb_logger, callbacks=[chkpt]) if DEVICE == "cuda" else Trainer(accelerator="cpu", logger=wandb_logger, callbacks=[chkpt])
    trainer.fit(
        model,
        datamodule=dm)
    
    return LightningFasterModule.load_from_checkpoint(chkpt.best_model_path)

if __name__ == "__main__":

    polyp_dm = PolypsPLDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    with wandb_context() as run:
        model = LightningFasterModule(CONFIG)
        if run is not None:
            run.watch(model)
        trained_model = train_model(model, run_name="head", dm=polyp_dm, run=run)

        model.full_train()
        model = train_model(model, run_name="full", dm=polyp_dm, run=run)