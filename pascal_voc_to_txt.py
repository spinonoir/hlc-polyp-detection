import os
import pathlib
import xml.etree.ElementTree as ET

#same number of annotations and images
#repeat for val and test
pathToAnnotationsTrain = "PolypsSet/train2019/Annotation/"
pathToAnnotationsTrainTxt = "PolypsSet/train2019/Annotation_txt/"
pathToAnnotationsVal = "PolypsSet/val2019/Annotation/"
pathToAnnotationsValTxt = "PolypsSet/val2019/Annotation_txt/"
pathToAnnotationsTest = "PolypsSet/test2019/Annotation/"
pathToAnnotationsTestTxt = "PolypsSet/test2019/Annotation_txt/"

def generate_txt(f, path_to_pascal_voc, path_to_txt):
    if f[-4:] == ".xml":
        tree = ET.parse(path_to_pascal_voc + f)
        root = tree.getroot()

        if not root.iter("object"):
            print("class not found")

        image_width, image_height = 0, 0
        for attr in root.iter("size"):
            image_width = int(attr.find("width").text)
            image_height = int(attr.find("height").text)
        
        bboxes = []
        for bbox in root.iter("bndbox"):
            bbox = [int(bbox.find("xmin").text),
              int(bbox.find("ymin").text),
              int(bbox.find("xmax").text),
              int(bbox.find("ymax").text)]
            bboxes.append(bbox)

        # create .txt file
        num = int(f[:-4])
        with open(path_to_txt + str(num) + ".txt", "w") as new_annotation:
            for bbox in bboxes:
                xcen = (bbox[0]+bbox[2])/2
                ycen = (bbox[1]+bbox[3])/2

                bbox_width = bbox[2]-bbox[0]
                bbox_height = bbox[3]-bbox[1]

                # normalize
                xcen /= image_width
                bbox_width /= image_width
                ycen /= image_height
                bbox_height /= image_height

                new_annotation.write("0 " + str(xcen) + " " + str(ycen) + " " + str(bbox_width) + " " + str(bbox_height) + "\n")

def convert_pascal_voc_to_txt(path_to_pascal_voc, path_to_txt, type):
  for f in os.listdir(path_to_pascal_voc):
      
      # only process files
      print(f) #directory name
      if type == "train":
          generate_txt(f, path_to_pascal_voc, path_to_txt)
      else:
          if f.isnumeric():
              path_to_pascal_voc_sequence = path_to_pascal_voc + f + '/'
              path_to_txt_sequence = path_to_txt + f + '/'
              os.mkdir(path_to_txt_sequence)
              for f_seq in os.listdir(path_to_pascal_voc_sequence):
                  generate_txt(f_seq, path_to_pascal_voc_sequence, path_to_txt_sequence)

# train - 28773 files; will take some time to convert
#os.mkdir(pathToAnnotationsTrainTxt)
#convert_pascal_voc_to_txt(pathToAnnotationsTrain, pathToAnnotationsTrainTxt, 'train')

# val
#os.mkdir(pathToAnnotationsValTxt)
#convert_pascal_voc_to_txt(pathToAnnotationsVal, pathToAnnotationsValTxt, 'val')

# test
#os.mkdir(pathToAnnotationsTestTxt)
#convert_pascal_voc_to_txt(pathToAnnotationsTest, pathToAnnotationsTestTxt, 'test')