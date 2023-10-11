from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2 import model_zoo
import cv2
import numpy as np

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load your custom config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

dataset_name = "plate_dataset"

# Define your custom class names
custom_class_names = ["Large_Plate", "Small_Plate"]

# Create a custom metadata dictionary
custom_metadata = {
    "thing_classes": custom_class_names,
}

# Add the custom metadata to the MetadataCatalog
MetadataCatalog.get(dataset_name).set(**custom_metadata)

# Access the metadata for the dummy dataset
metadata = MetadataCatalog.get(dataset_name)

# Load your custom model weights
cfg.MODEL.WEIGHTS = "platecount/model_final.pth"
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.CLASSES = ["small_plate", "large_plate"]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Create a custom model
model = build_model(cfg)

# Create a predictor using your custom model
predictor = DefaultPredictor(cfg)

# Load your image
image_path = "platecount/test/output2808_new_jpg.rf.1d7a3302a61ea5fd48353d71c6a7d825.jpg"
im = cv2.imread(image_path)

# Perform inference
outputs = predictor(im)
print('output : ', outputs['instances'].pred_boxes.tensor.cpu().numpy())
Boxes = outputs['instances'].pred_boxes.Boxes.tensor.cpu().numpy()
labels = outputs['instances'].pred_classes.tensor.cpu().numpy()

# Visualize the results
class_names = ['small_plate', 'large_plate']
v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Object Detection Results", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
