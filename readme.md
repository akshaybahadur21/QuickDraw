## FaceEncoder
This code helps to undertand the concept of Autoencoders. The autoencoder is trained on my facial dataset and it learns encoding of 1024 units from an input of 7500 pixels. This is a two step procedure.
1) Encoder - which learns embedding from the input dimensions.
2) Decoder - which recreates the image from the embedding created by the encoder.

#### Types of Autoencoders used
1) Simple Network
2) Deep Network
3) Convolutional Network

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction. Recently, the autoencoder concept has become more widely used for learning generative models of data.

### Dataset
Creating our own dataset

### Python  Implementation

1) Network Used- Simple Network, Deep Network, Convolutional Network
2) Technique - Autoencoders

If you face any problem, kindly raise an issue

### Procedure

1) First run `LoadData.py` which will load the images from folder `1` (you can change the name) and store it into a pickle file.
2) Now, run `FaceCoder.py` which will train a simple, deep and a convolutional autoencoder and store it in h5 file.
2) Now you need to have the data, run `FaceApp.py` which will use dlib library to get your face, encodes it and then decodes it to display the image.
3) For altering the model, check `FaceCoder.py`.
4) For tensorboard visualization, go to the specific log directory and run this command ` tensorboard --logdir=.` You can go to `localhost:6006` for visualizing your loss function.

<img src="https://github.com/akshaybahadur21/FaceEncoder/blob/master/faceEnc.gif">

### References:
 
 - [Building Autoencoders in Keras - The Keras Blog](https://blog.keras.io/building-autoencoders-in-keras.html) 





