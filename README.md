# Speech Analysis and Wav2Vec2 Feature Extraction

### Final Year Project | Python | Speech Signal Processing | Deep Learning

## 📌 Project Overview

Speech Analysis and Wav2Vec2 Feature Extraction is a Python-based desktop application developed to analyse speech signals and extract deep learning-based speech features from audio files.

The application allows users to upload WAV audio files, visualize speech signals through waveform and spectrogram representations, reduce background noise, and extract meaningful speech features using the pretrained Wav2Vec2 model.

The extracted features are saved in CSV format for further processing and analysis.

## 🎯 Project Objectives

- To analyse speech signals using digital signal processing techniques.
- To visualize audio signals through waveform and spectrogram representations.
- To reduce background noise from speech recordings.
- To extract deep learning-based speech features using Wav2Vec2.
- To save extracted features in CSV format for further analysis.

## ✨ Key Features

- **Audio File Loading:** Upload and process WAV audio files.
- **Waveform Visualization:** Display the amplitude variations of speech signals over time.
- **Spectrogram Visualization:** Analyse the frequency components of speech signals.
- **Color and Grayscale Modes:** Switch between different spectrogram display modes.
- **Noise Reduction:** Reduce background noise using the Noisereduce library.
- **Wav2Vec2 Feature Extraction:** Extract deep learning-based speech representations.
- **Mean and Standard Deviation Pooling:** Generate a 1,536-dimensional feature vector.
- **CSV Export:** Save extracted speech features into CSV files.

## 🛠️ Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Tkinter | Desktop graphical user interface |
| NumPy | Numerical computations |
| SciPy | Audio signal processing |
| Matplotlib | Waveform and spectrogram visualization |
| Librosa | Audio loading and resampling |
| NoiseReduce | Background noise reduction |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Wav2Vec2 model integration |
| Pandas | CSV data handling |

## 🧠 Wav2Vec2 Model

This project uses the pretrained `facebook/wav2vec2-base` model to extract speech representations.

The audio is processed at a sampling rate of 16 kHz and converted into deep learning feature representations.

The project extracts:
- 768-dimensional mean-pooled features.
- 768-dimensional standard deviation-pooled features.

These are combined to generate a **1,536-dimensional feature vector** for each audio file.

## ⚙️ Project Workflow

1. Upload a WAV audio file.
2. Load and process the audio signal.
3. Visualize the waveform representation.
4. Generate and display the spectrogram.
5. Apply noise reduction during feature extraction.
6. Process the audio using the Wav2Vec2 model.
7. Extract mean and standard deviation features.
8. Export the extracted features into a CSV file.

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Speech-Analysis-Wav2Vec2.git
```

### 2. Navigate to the Project Directory

```bash
cd Speech-Analysis-Wav2Vec2
```

### 3. Install Required Libraries

```bash
pip install numpy scipy matplotlib librosa noisereduce pandas torch transformers
```

### 4. Run the Application

```bash
python final.py
```

**Note:** An internet connection may be required the first time the application loads the pretrained Wav2Vec2 model. A compatible Python environment and the required library dependencies are necessary.

## 📂 Project Structure

```text
Speech-Analysis-Wav2Vec2/
│
├── final.py
├── README.md
├── requirements.txt
│
└── screenshots/
    ├── main_interface.png
    ├── waveform.png
    ├── color_spectrogram.png
    └── grayscale_spectrogram.png
```

## 📊 Output

The application generates a CSV file containing:
- Audio filename
- Sampling rate
- Audio duration
- 1,536 extracted Wav2Vec2 features

The generated CSV file can be used for further analysis and machine learning experiments.

## 🚀 Future Enhancements

- Add support for additional audio file formats.
- Improve the graphical user interface.
- Integrate speech classification techniques.
- Develop additional speech analysis capabilities.
- Explore real-time speech feature extraction.

## 👩‍💻 Project Information

**Project Title:** Speech Analysis and Wav2Vec2 Feature Extraction

**Project Type:** Final Year Academic Project

**Programming Language:** Python

**Author:** YOUR NAME

**GitHub:** YOUR GITHUB PROFILE LINK

---

⭐ If you find this project interesting, feel free to explore the repository!
