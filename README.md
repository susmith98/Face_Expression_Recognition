# Face_Expression_Recognition
Face Expression Recognition Model built using Transfer Learning(Feature extraction) from VGG pretrained model

Overview of What's happening in Edetector.py

1) Loading a pretrained VGG model from keras.applications
2) Creating a VGGtop Custom Model on top of VGG Model
3) Feature Extraction of Training and Validation data by passing them through VGG model
4) Training VGGtop Model with Extracted Features 
5) Using Both the Models Predicting facial expressions of multiple faces in frame using libraries cv2,dlib
