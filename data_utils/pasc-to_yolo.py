import pandas as pd
import os

def convert_box(bbox, img_shape):
    """
    Converts PASCAL_VOC bbox to YOLO Darknet txt string
    Input: bbox = [xmin, ymin, xmax, ymax], img_shape = [width, height]
    Output: class (always 0 for polyp), x_center, y_center. width, height 
        normalized
    """
    x_center = ((bbox[0] + bbox[2]) / 2) / img_shape[0]
    y_center = ((bbox[1] + bbox[3]) / 2) / img_shape[1]
    box_w = (bbox[2] - bbox[0]) / img_shape[0]
    box_h = (bbox[3] - bbox[1]) / img_shape[1]
    label = f"0, {x_center}, {y_center}, {box_w}, {box_h}"
    return label

if __name__ == main(): 
    df = pd.read_csv("all_labels_yolo.csv")
    df = df[df['label'] == 1]

    for _, row in df.iterrows():
        label_path = f"{row['path'][:-4]}.txt"
        bbox = [df['xmin'], ]df['ymin'], df['xmax'], df['ymax']]
        img_shape = [df['width'], df['height']]
        label_txt = convert_box(bbox, img_shape)
