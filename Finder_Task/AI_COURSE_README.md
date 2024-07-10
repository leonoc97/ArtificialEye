**Goal of the project:**
Develop an object detection algorithm for the **Cybathlon Vision Task "Finder"**.

**Approach of the project:**
Compare a model trained with **Roboflow** with a **Custom-trained YOLOv9 model**

Several videos were recorded of all six objects that are included in the Cybathlon task from different perspectives and with varying arrangement of objects on the floor.
Frames were than extracted from the videos for the generation of the dataset and later training of the model. 
The preprocessing of the dataset was done completely in Roboflow, including labeling of the images and augmentations.

The finished dataset as well as the Roboflow model and the Custom-trained model (.ipynb) are included in this Repository.
To prove the correct predictions of the Roboflow model, the model is applied to a testing image containing all items to be detected.

- Dataset: Folder "**Annotated images**" <br>
- Roboflow model: Folder "Cybathlon_Roboflow_model" -> File "**Roboflow_model.ipynb**" <br>
- Custom Yolov9 model: File "**Finder_Task_YOLO9_training.ipynb**"
