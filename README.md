# Count_Object_Dishes
The objective is to count the plates using object detection and object tracking. The repository used the PyimageSearch Simple Object Tracking code frome [here](https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/). The object detection model used is pre-trained Faster R-CNN model trained on Detectron2 framework and a custom dataset.



# Output of the code counting the dishes!

![ezgif com-crop (1)](https://github.com/skp-1997/Count_Object_Dishes/assets/97504177/fb932ec8-bb5a-4111-bd09-48dd731c3d2c)


# Installation of the environment

We encourage you to use conda environment. Once you create an environment use follwoing command to get the environment ready

```
pip install -r requirements.txt
```

# Training the Faster R-CNN Model

Follow instruction given in PlateCount_FasterRCNN.ipynb file for training the model. You can use the data from the roboflow account provided in the file.

# Inference on Images and Videos

Run the command to test on an Image

```
python test_images2.py
```

Run the command to test on a Video

```
python test_video.py
```

Make sure to provide right path of video, model file and image file in the script.

# Common Debug

1. Make sure the environment is properly installed
2. Provide complete path of the model file
3. Detectron2 installation will take time. Be patient.

