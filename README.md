
# Face-Recognition-Deep-Learning-Project

## Problem Statement

We need a solution where in model is not only able to detect the face whether its of human(s) or animal(s), it should also be able to identify their names.
If we have 4 person and 2 animals (Cat,dog) whose faces gets clicked by the model and is stored in the system with the names of each person and the animals then once training happens model should be able to identify the person and the animals
as soon as their faces appear on the camera.


## Solution Proposed

In this project, the focus is to correctly detect the face and identify the face of the users/animals using deepinsight/InsightFace.
InsightFace is an integrated Python library for 2D&3D face analysis. It efficiently implements a rich variety of state of the art algorithms of face recognition, face detection and face alignment, which optimized for both training and deployment. </br>
**github link of InsightFace** https://github.com/deepinsight/insightface

## Tech Stack Used
1. Python </br>
2. MTCNN(Multi-task Cascaded Convolutional Networks)  https://pypi.org/project/mtcnn/
3. Keras to train the model </br>
  


# How to run the project
**Step 1** : open your anaconda prompt (for windows user search inside start menu )
                                   (for Ubuntu and Mac user you can open your terminal)

**Step 2** : Create a new environment
                command : conda create -n facerecognition python==3.6.9 -y </br>
                
**Step 3** : activate your environment  </br>
                conda activate facerecognition  </br>
**Step 4** : conda install -c anaconda mxnet </br>

**Step 5** : conda install -c conda-forge dlib </br>

**Step 6** : Uninstall existing version of numpy and install numpy 1.16.1 version: </br>
        pip uninstall numpy </br>
        pip uninstall numpy </br>
        pip install numpy==1.16.1 </br>

**Step 7**:  Install requirements.txt in the newly created environment</br>
         pip install -r requirements.txt</br>

**Step 8** : Installation and setup is done:</br>
         a).  cd src</br>
         b). python app.py</br>


## Video link of project demo
https://youtu.be/MKOaQu3aXSs

## How project was designed and build
1. **app.py->** Driver program of the project which invokes the camera and then call subsquent method from each modules to perform the operations of collecting pictures from camera,training it and prediction of the face . </br>
2. **get_faces_from_camera.py->** Purpose is the get the 50 images from live feed of camera and crop the facial feature of the image and save it in 112 * 112 dimension </br> 
3. **faces_embedding.py->** Purpose of this class is to convert image into numerical value and saving it in pickel format. This process is called Face Embedding </br>
4. **train_softmax.py->** Purpose is to train the model using embeddings of the image. Model is trained in batchsize of 8 with 5 epochs. Relu activation for hidden layer and softmax for output layer. Saving the output as pickle format.</br>
5. **facePredictor.py->** Purpose is to do the prediction of the face. </br>



## Logic behind Face Recongition Technique
1. Get input images of the human faces. </br>
2. Human faces needs to be labelled with the name. </br>
3. Input image of size 1280 * 720 needs to be cropped to size of 96 * 96 or 128 * 128 and then feed to deep learning algorithm.
4. MTCNN detects the bounding box co-ordinates, co-ordinates of keypoints of the face(nose, mouth-right,right-eye,left-eye,mouth_left) and the confidence score of the face image)
5. Then we need to do the Facial analysis for which we need to create small feature and create array of the features.
6. We need to convert the image data into the numbers also called **Embeddings.** </br> ![image](https://github.com/ravi0dubey/Face-Recognition-Deep-Learning-Project/assets/38419795/3b1032e8-b053-46a4-9d4a-028a40ed705c) </br>
7. Using Embeddings of the image(s) we can choose either **Machine learning**, **Deep Learning** or the **Distance approach**(Cosine Distance and Consine Similarity) to do the facial recognition.s </br>
8. In case of cosine similarity threshold is set to **.8**.
9. If in case an unknown person/animal face comes up during prediction whose image has not been trained in such case model will show as **unknown**. </br>
10. We use **tracking** to stop processing the Face recognition if the face remains the same during live feed i.e no new faces comes up and the existing face has already been recognized. This is done to minimize the computation done for already identified face.</br>
11. If no face is found for recognition, we need to stop tracking to minimize the computation. </br>



## Training and Accuracy Loss
![accuracy_loss](https://github.com/ravi0dubey/Face-Recognition-Deep-Learning-Project/assets/38419795/fd1a5d61-6e4f-4930-bfce-3d60c8aa91ba)



