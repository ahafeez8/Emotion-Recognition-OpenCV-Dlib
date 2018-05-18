# Realtime-Emotion-Recognition-OpenCV-Dlib
we have trained our model to determine the emotions in the same pattern using a technique called Machine Learning. ML as its name denotes, is the process making an otherwise oblivious, computer learn something to predict in a real time solution. It is almost alike teaching a student how to solve a particular problem and the learned student is then able to solve test problems he is given.

This program can be run on both Linux and Windows, installation of above packages however differ on both. It is coded in Computer Programming Language, Python version 3.6. 

Same have we done in this model. This program does real time emotion recognition using Machine Learning algorithm called Linear SVM. There are three aspects to this:

### Pre-processing of data
We use a part of CK dataset which is labelled on 7 emotions: Anger, disgust, joy, neutral, sadness, fear and surprise. 
A human face has 68 landmark points of movement which can help us determine the emotion. This done through change in positions of these landmarks and they are fed as features to our training model. 

<p align="left">
  <img src="https://www.researchgate.net/profile/Sebastien_Marcel/publication/37434867/figure/fig1/AS:309878666088448@1450892239254/Example-face-image-annotated-with-68-landmarks.png" width="250" height="270"/>
</p>
Fig: Facial Landamrks
Source:https://www.researchgate.net/figure/Example-face-image-annotated-with-68-landmarks_fig1_37434867

To determine change in these landmarks, we use a center point on face and we determine changes in contrast to that point. 

### Training our model on labelled dataset.
Then we train our Machine learning model based on this processed dataset. We extract the labels of each image and feed image landmarks as features and train them against the labels. This, in a nutshell means our system learning what positions of facial landmarks corresponds to which emotion.

### Testing our model and determine scores. 
Use a chunk of the same dataset, we test our model and record scores of each. Our system does 7 epochs and uses a linear SVM to train a dataset of approximately 500 images. 

<p align="left">
  <img src="https://drive.google.com/file/d/1bmLG9BtfHjxkyOXG3El_-GU0vFlRHZJ-/view?usp=sharing"/>
</p>

### Run the model for real time feed. 
The program takes input from a webcam and displays on console the predicted emotion as it detects in real time. Some example snippets are attached in res1.png and res2.png. 
