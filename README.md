# Audio Classification System

This project is an **Audio Classification System** that allows users to classify audio files as either "music" or "speech" using machine learning models. The system is built using Java Swing for the graphical user interface (GUI) and the WEKA library for machine learning.



## Features

- **Audio Playback**: Play and stop audio files directly from the GUI.
- **Feature Extraction**: Extract audio features such as:
  - Maximum amplitude
  - Root Mean Square (RMS) energy
  - Zero-crossing rate
  - Spectral centroid
  - Spectral bandwidth
  - Spectral rolloff
- **Model Selection**: Choose between the following classifiers:
  - Decision Tree (J48)
  - Logistic Regression
- **File Classification**: Classify a selected audio file and display the result.
- **Model Details**: Perform 10-fold cross-validation and display model evaluation metrics such as precision, recall, accuracy, F1-score, and confusion matrix.

## System Requirements

- **Java Development Kit (JDK)**: Version 8 or higher
- **WEKA Library**: Included in `libs/weka.jar`
- **JTransforms Library**: Included in `libs/JTransforms-3.1-with-dependencies.jar`
- **Commons Math4 Library**: Included in `libs/commons-math4-core-4.0-beta1.jar`
- **Audio Files**: `.wav` format organized in `audio/music` and `audio/speech` directories

## Setup and Installation



1. **Compile the Code**:
   ```bash
   javac -cp .:libs/weka.jar:libs/JTransforms-3.1-with-dependencies.jar:libs/commons-math4-core-4.0-beta1.jar src/AudioPlayerGUI.java
   ```

2. **Run the Application**:
   ```bash
   java -cp .:libs/weka.jar:libs/JTransforms-3.1-with-dependencies.jar:libs/commons-math4-core-4.0-beta1.jar src.AudioPlayerGUI
   ```

## File Structure

```
.
├── src
│   ├── AudioPlayerGUI.java  # Main application file
├── libs
│   ├── weka.jar             # WEKA library
│   ├── JTransforms-3.1-with-dependencies.jar  # FFT library
│   ├── commons-math4-core-4.0-beta1.jar       # Math utilities
├── audio
│   ├── music                # Directory for music audio files
│   ├── speech               # Directory for speech audio files
├── features.csv             # Extracted features (auto-generated)
├── features.arff            # ARFF file for WEKA (auto-generated)
```

## How to Use

1. **Load Audio Files**: Ensure `.wav` files are placed in the `audio/music` and `audio/speech` directories.
2. **Start the Application**: Run the compiled Java program.
3. **Select a Model**: Choose either `Decision Tree (J48)` or `Logistic Regression` from the dropdown.
4. **Play Audio**: Select an audio file from the list and click `Play Selected File`.
5. **Classify Audio**:
   - Select an audio file.
   - Click `Classify Selected File`.
   - View the classification result in a popup.
6. **View Model Details**:
   - Click `Get Model Details`.
   - View model evaluation metrics in a popup.

## Key Functions

### 1. Audio Playback
- **Play Selected File**: Plays the selected audio file.
- **Stop Audio**: Stops the currently playing audio.

### 2. Feature Extraction
The following features are extracted from each audio file:
- Time-domain features:
  - Maximum amplitude
  - RMS energy
  - Zero-crossing rate
- Frequency-domain features (via FFT):
  - Spectral centroid
  - Spectral bandwidth
  - Spectral rolloff

### 3. Classification
- **Decision Tree (J48)**: A simple and interpretable tree-based model.
- **Logistic Regression**: A linear classification model.
- Classification results are displayed in a popup.

### 4. Model Evaluation
- **Metrics**:
  - Precision
  - Recall
  - Accuracy
  - F1-Score
- **Confusion Matrix**: Provides detailed classification results.
- Results are displayed in a popup for easy review.

## Example Popups

### Classification Result
```plaintext
The file "audio1.wav" is classified as: music
```

### Model Details
```plaintext
Results for Decision Tree (J48):
Precision: 0.92
Recall: 0.90
Accuracy: 91.8%
F1-Score: 0.91

Summary:
Correctly Classified Instances   37               92.5%
Incorrectly Classified Instances  3                7.5%

Class Details:
...

Confusion Matrix:
...
```

## Known Issues

1. The application currently supports only `.wav` files.
2. Audio files must be placed in the correct directories (`audio/music` and `audio/speech`).
3. Ensure proper formatting of `features.csv` to avoid errors during ARFF conversion.

## Future Enhancements

- Add support for additional audio formats.
- Enhance feature extraction with more advanced audio analysis techniques.
- Integrate more machine learning models.
- Provide visualizations for classification results.

## Acknowledgments

- **WEKA**: Machine Learning toolkit
- **JTransforms**: Fast Fourier Transform library
- **Apache Commons Math**: Mathematical utilities library




## Developed By:
Talha Zain.
If you have any queries, you can contact me on my email talha.10.zain@gmail.com
