# Real_Time_Emotion_Detection
![Emotion Detection]((https://github.com/shivam8764/Real_Time_Emotion_Detection))

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Audio Feedback](#audio-feedback)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
Welcome to the **Real-Time Emotion Detection** project! This application captures live video from your webcam, analyzes facial expressions, and accurately predicts the emotions being displayed in real time. By leveraging advanced deep learning techniques and integrating audio feedback, this project aims to provide an interactive and seamless user experience.

## Features
- **Real-Time Emotion Analysis**: Detects and classifies emotions from live video feed.
- **Deep Learning Powered**: Utilizes a Convolutional Neural Network (CNN) trained on the FER-2013 dataset for accurate predictions.
- **Interactive Audio Output**: Integrates Pyttsx3 to provide voice-based feedback of detected emotions.
- **User-Friendly Interface**: Simple and intuitive interface using OpenCV for seamless interaction.

## Technologies Used
- **Python**: Programming language used for the entire project.
- **Keras**: High-level neural networks API for building and training the CNN model.
- **OpenCV**: Open Source Computer Vision Library for real-time video capture and processing.
- **Pyttsx3**: Text-to-speech conversion library for audio feedback.
- **TensorFlow**: Backend for Keras to handle deep learning computations.

## Installation
Follow these steps to get the project up and running on your local machine.

### Prerequisites
- Python 3.6 or higher
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Real-Time-Emotion-Detection.git
   cd Real-Time-Emotion-Detection
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application using the following command:
```bash
python app.py
```
Ensure your webcam is connected and functional. The application will open a window displaying the live video feed with detected emotions annotated on the screen. Additionally, you will hear the detected emotion through audio feedback.

## Dataset
This project utilizes the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset, which consists of 35,887 labeled facial images categorized into seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model Training
To train the CNN model on the FER-2013 dataset, follow these steps:

1. **Download the Dataset**
   - Ensure the FER-2013 dataset is downloaded and placed in the `data/` directory.

2. **Run the Training Script**
   ```bash
   python train_model.py
   ```
   This script preprocesses the data, builds the CNN architecture, and trains the model. After training, the model weights are saved for future use.

## Audio Feedback
The project integrates **Pyttsx3** to provide audio feedback for detected emotions. This enhances user interaction by verbally announcing the detected emotion, making the experience more engaging.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or suggestions, feel free to reach out:

- Email: shivam8764@gmail.com
- GitHub: https://github.com/shivam8764/

---

Thank you for checking out the Real-Time Emotion Detection project! Feel free to explore, use, and contribute to enhance its capabilities.
