# SignBuddy - Sign Language Detection

SignBuddy is a real-time sign language recognition web application built using Flask, OpenCV, and Deep Learning.

The project detects hand gestures using a webcam and converts sign language gestures into text and voice output.

## Features

- Real-time hand gesture detection
- Phrase recognition mode
- Number recognition mode
- Voice output using text-to-speech
- Dark mode user interface
- Responsive web design

## Technologies Used

- Python
- Flask
- OpenCV
- CVZone
- MediaPipe
- TensorFlow / Keras
- HTML
- CSS
- JavaScript

## How I Built This Project

- Learned basic sign language gestures and hand signs
- Collected hand gesture image data with the help of classmates and relatives
- Organized images into labeled folders for each gesture
- Trained the deep learning model using TensorFlow and Google Colab
- Retrained the model with additional datasets to improve accuracy
- Designed the frontend using HTML, CSS, and JavaScript
- Created a Flask backend for real-time prediction
- Integrated the trained model into the web application
- Added text-to-speech functionality for voice output

## How to Run

1: Install required packages:

pip install -r requirements.txt

2: Run the application:

python signbuddy.py

3: Then open:

http://127.0.0.1:5000/

## Machine Learning

- Used MobileNetV2 for transfer learning
- Applied data augmentation
- Trained and retrained custom gesture recognition models

## Future Improvements

- Add more sign language gestures
- Improve model accuracy
- Deploy project online
- Add sentence generation support

## Author

Ankita Pandey
