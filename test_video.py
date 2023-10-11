import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

from pyimagesearch.centroidtracker import CentroidTracker

import numpy as np
from tqdm import tqdm

# Load your custom configuration file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.merge_from_file("path/to/your/config.yaml")

dataset_name = "plate_dataset"

# Define your custom class names
custom_class_names = {0 : "Large_Plate", 1 : "Small_Plate"}

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



# Create a predictor using your custom model and configuration
predictor = DefaultPredictor(cfg)

video = cv2.VideoCapture('input/record_3_small_plates.mp4')

# Get video properties for VideoWriter
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

videowriter = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
small_plates = 0
large_plates = 0

ct = CentroidTracker()

# Initialize tqdm for the progress bar
with tqdm(total=total_frames, desc="Processing") as pbar:
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Perform object detection on the frame
        outputs = predictor(frame)

        # Extract the instances (bounding boxes and labels)
        instances = outputs["instances"].to("cpu")

        # Loop through detected instances and draw bounding boxes
        bboxes = []
        labels = []
        for i in range(len(instances)):
            box = instances.pred_boxes.tensor[i].detach().numpy().astype(int).tolist()
            x1,y1,x2,y2= max(box[0],0),max(box[1],0),min(box[2],width),min(box[3],height)
            label = custom_class_names[int(instances.pred_classes[i].item())]
            score = instances.scores[i].item()

            # Draw a green bounding box with label and score
            color = (0, 0, 255)  # Green color in BGR
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_patch = np.zeros((label_height + baseline, label_width, 3), np.uint8)
            label_patch[:,:] = (0,255,255)
            labelImage= cv2.putText(label_patch, label, (0, label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            label_height,label_width,_= labelImage.shape
            if y1-label_height< 0:
                y1=label_height
            if x1+label_width> width:
                x1=width-label_width
            frame[y1-label_height:y1,x1:x1+label_width]= labelImage
            bboxes.append(box)
            labels.append(label)
            '''
            if box[1] > height//2:
                if label == 'Small_Plate':
                    small_plates += 1 
                else:
                    large_plates += 1
            '''
            plate_count = ct.getObjectCount()

            cv2.putText(frame, f"Plate Count: {plate_count}", (width//5, height//10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

        
        objects = ct.update(bboxes)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

        # result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        frame_count += 1
        pbar.update(1)  # Update tqdm progress bar
        videowriter.write(frame)


video.release()
videowriter.release()

