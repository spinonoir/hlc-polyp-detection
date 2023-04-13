# hlc-polyp-detection
Hidden Layer Cake - Computer Vision for Colonoscopy Polyp Detection 

## Faster RCNN Polyp Detection Module

### Extracting FasterRCNN data directory from data.zip:
***Make sure the data dir is in `.gitignore`!!!***
+ unzip contents and place in the same directory as `FasterRCNN_working.ipynb`
+ make sure the path is correct: 
+ should match the diagram below, not `./data/data`
```
data/
    ├── test/
    │   ├── test_labels.csv
    │   │   
    │   └── images/
    │       ├── test_seqM_frameN.jpg
    │       ├── ...
    ├── train/
    │   ├── train_labels.csv
    │   │ 
    │   └── images/
    │       ├── train_seqM_frameN.jpg
    │       ├── ... .jpg  
    └── val/
        ├── val_labels.csv
        │  
        └── images/
            ├── val_seqM_frameN.jpg
            ├── ... .jpg  
```

### Training a model Train/Val sets:
+ `FasterRCNN_working_executed.ipynb` is the first notebook that was trained on the data set using a paperspace cloud compute instance
+ Clone this notebook and give it a new, meaningful name
+ see `./src/config.py` for info on default values passed around
+ Follow directions in markdown cells inside the notebook
+ The model will save it's weights every 2 Epochs (defined in `src/config.py`) to `./output/%MODEL_NAME_%EPOCH.pth`
  + giving the model a meaningful name will make it easier to sort through the different weights files later. 
+ Currently the plots don't work -- idk why
+ when the model is finished training, move the best performing model weights to `./weights/other_weights` and update the `README.md` in that directory with the information about how that model was trained. 
### Finetuning the Model
 The model is "defined" in `src/model.py`, but it returns a standard PyTorch model, so the weights can be updated upon return
  + Initially configured with `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")` as the model. To make changes to the backbone either create a new fucntion in `./src/model.py` or comment out what's there and add your own. 
  + TODO: convert function in `model.py` to a class that inherits from torch and allow for choice of backbone
  + Image augmentations with `albumentations` are configured in `./src/utils.py:get_transformations()` you can update the training transformations with more complex transforms using the `albumentations` library.
    + see their documentation for more info https://albumentations.ai/docs/ 
    + we rely on this library to ensure that the bounding boxes are also transformed, so changing this library will require an understanding of the bounding box label format. 
  
### Evaluating the Model
+ use the `FasterRCNN_evaluation.ipynb` notebook to run through the test data and calculate various metrics
+ STILL UPDATING README, but the notebook is well documented

## Data Utils 
- contains some scripts for processing PASCAL_VOC xml labels into CSV files.

## Scratch 
- scraps [jarret] didn't want to delete yet. 