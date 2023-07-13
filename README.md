# Face App
Face App is a Python-based application that uses TensorFlow for machine learning. This app can detect your face, track your iris, and try to predict your age and gender (although not very accurate for the last two). The
demonstration for this program can be found here! [Click me for the video!](https://youtu.be/zvNkCcAA7Dw)

## Things I Used: 
- OpenCV: using my webcam and allowing me to implement my models
- Kaggle: datasets and also GPUs to train my models
- Tensorflow: machine learning
- [Labelme](https://github.com/wkentaro/labelme): annotating personal images
- [Albumentations](https://albumentations.ai/): augmenting and upscaling my dataset 

## Face Detection
For this model, I used my own personal dataset and took about 100 images using **OpenCV**, which I then annotated the bounding boxes for each image using **Labelme**. Finally I used Albumentations 
to upsasmple and augment my dataset since having more than 100 images would help make my model better. I split the data into a training, test, and validation set for when I train the model. To train 
the model, I used Kaggle and used a jupyter notebook, as it would be easier to debug and test.

For this model, I used transfer learning by using the **VGG16** model and added some **Dense** layers.  

The training file for this can be found in the "training_files" at "face_detect_train.ipynb".

## Iris Tracker
For this model, I did almost the exact same thing as face detection, except instead of bounding boxes, I used you points or dots as my labels. The rest of the steps were almost identical.

This model also used transfer learning, but with the **ResNet152** model, where I added extra **Convolution** layers and some dropout to help with overfitting.

The training file for this can be found in the "training_files" at "iris_train.ipynb".

## Age and Gender Prediction
I used a kaggle dataset for this model, more specifically the UTKFace dataset. It included over 20000 images of people's faces with ranging ages from 0-116, different genders, and much more. I only needed the gender and 
ages, so I extracted those out of the images. However when training the model, there was some overfitting and the model didn't end up perfect, and it can be visible when you watch the video of the program. I am 18 but it 
is predicted a completely different value. Some errors that may have happened could be during the preprocessing stage, and I may have not done it properly. 

For this model I used the **VGG16** Model with some **Dense** layers.

# Final Thoughts
Overall, this project was fun to make! I think it turned out better than I expected, and it was able to accurately track my face and iris! Here are some things I could add or improve on in the future: 
Things to add: 
- Emotion detection
- Lip Reader
- Drowsiness Detection

Things to improve on:
- Fix the issues with age and gender detection
- Rebuild the iris dataset so that it can track when only one iris is open (it currently only works when both iris are open)
