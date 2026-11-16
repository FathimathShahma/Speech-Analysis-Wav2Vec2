from tkinter import *
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import torchaudio
import noisereduce as nr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, T5ForConditionalGeneration, T5Tokenizer
import os
import librosa

# ==============================
# Load Pretrained Wav2Vec2 Model
# ==============================
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# ==============================
# Load Grammar/Correction Model
# ==============================
CORRECT_MODEL = "t5-base"
correct_tokenizer = T5Tokenizer.from_pretrained(CORRECT_MODEL)
correct_model = T5ForConditionalGeneration.from_pretrained(CORRECT_MODEL)

def correct_text(raw_text):
    """Correct transcription using T5"""
    input_text = "fix grammar: " + raw_text
    inputs = correct_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = correct_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    corrected = correct_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# ==============================
# Tkinter Window Setup
# ==============================
root = Tk()
root.title('SPEECH Wav2Vec2 Recognition with Correction')
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f'{screen_width}x{screen_height}')
try:
    root.state('zoomed')
except Exception:
    pass
root.configure(background='light gray')

# Canvases
c1 = Canvas(root, bg='gray', width=screen_width-25, height=screen_height-105)
c1.place(x=10, y=10)
c2 = Canvas(root, bg='white', width=screen_width-45, height=220)  # waveform
c2.place(x=20, y=50)
c3 = Canvas(root, bg='white', width=screen_width-45, height=220)  # spectrogram
c3.place(x=20, y=300)

# Status bar
status_var = StringVar(value="Load a WAV file (File → Open) or Record your voice")
status_bar = Label(root, textvariable=status_var, anchor='w', bg='#eee')
status_bar.place(x=20, y=660, width=screen_width-40, height=26)

# Transcript box
transcript_box = Text(root, height=8, wrap="word", bg="white", font=("Arial", 12))
transcript_box.place(x=20, y=600, width=screen_width-40, height=80)

# Recording Variables
is_recording = False
recorded_audio = None
fs = 16000  # sampling rate

# ==============================
# Functions
# ==============================
def plot_waveform_and_spectrogram(data, sr):
    """Plot waveform and spectrogram in Tkinter canvases"""
    for widget in c2.winfo_children():
        widget.destroy()
    for widget in c3.winfo_children():
        widget.destroy()

    dpi = 100
    # Waveform
    fig1 = Figure(figsize=(c2.winfo_width() / dpi, c2.winfo_height() / dpi),
                  dpi=dpi, tight_layout=True)
    ax1 = fig1.add_subplot(111)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time [s]')
    ax1.set_title("Voice Signal")
    ax1.grid(True)
    times = np.arange(len(data)) / sr
    ax1.plot(times, data, color='black')
    canvas1 = FigureCanvasTkAgg(fig1, master=c2)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill='both', expand=True)
    plt.close(fig1)

    # Spectrogram
    fig2 = Figure(figsize=(c3.winfo_width() / dpi, c3.winfo_height() / dpi),
                  dpi=dpi, tight_layout=True)
    ax2 = fig2.add_subplot(111)
    Pxx, freqs, bins, im = ax2.specgram(data, Fs=sr, cmap="viridis")
    ax2.set_title("Spectrogram")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    fig2.colorbar(im, ax=ax2).set_label("Intensity [dB]")
    canvas2 = FigureCanvasTkAgg(fig2, master=c3)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig2)



def recognize_speech(file_path):
    """Run noise reduction + wav2vec2 recognition + correction"""
    # Load audio safely with librosa
    speech_np, sr = librosa.load(file_path, sr=16000, mono=True)  # always 16kHz mono

    # Noise reduction
    reduced_noise = nr.reduce_noise(y=speech_np, sr=sr)

    # Recognition (Wav2Vec2)
    input_values = processor(reduced_noise, sampling_rate=sr,
                             return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    raw_transcription = processor.batch_decode(predicted_ids)[0]

    # Correction (T5)
    corrected_transcription = correct_text(raw_transcription)

    # Show results
    transcript_box.delete("1.0", END)
    transcript_box.insert(END, f"Raw: {raw_transcription}\n\nCorrected: {corrected_transcription}")


def sinput():
    """Browse file, plot waveform, spectrogram, then recognize"""
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return
    status_var.set(f"Loaded file: {os.path.basename(file_path)}")
    sr, data = wav.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # convert to mono
    plot_waveform_and_spectrogram(data, sr)
    recognize_speech(file_path)

def start_recording():
    global is_recording, recorded_audio
    if is_recording:
        messagebox.showinfo("Info", "Recording is already in progress.")
        return
    is_recording = True
    recorded_audio = []
    def _record():
        global recorded_audio
        status_var.set("Recording... Speak now.")
        while is_recording:
            data = sd.rec(int(1 * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            recorded_audio.append(data)
        status_var.set("Recording stopped. Use 'Save Recording' to save.")
    threading.Thread(target=_record, daemon=True).start()

def stop_recording():
    global is_recording
    if not is_recording:
        messagebox.showinfo("Info", "No recording is in progress.")
        return
    is_recording = False
    status_var.set("Recording stopped. Use 'Save Recording' to save.")

def save_recording():
    global recorded_audio
    if recorded_audio is None or len(recorded_audio) == 0:
        messagebox.showwarning("Warning", "No recorded audio to save.")
        return
    audio = np.concatenate(recorded_audio, axis=0)
    save_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV files", "*.wav")],
                                             title="Save Recording As")
    if save_path:
        wav.write(save_path, fs, audio)
        messagebox.showinfo("Saved", f"Recording saved at:\n{save_path}")
        status_var.set(f"Recording saved: {save_path}")
        plot_waveform_and_spectrogram(audio, fs)
        recognize_speech(save_path)

def play_recording():
    global recorded_audio
    if recorded_audio is None or len(recorded_audio) == 0:
        messagebox.showwarning("Warning", "No recorded audio to play.")
        return
    audio = np.concatenate(recorded_audio, axis=0)
    status_var.set("Playing recorded audio...")
    sd.play(audio, fs)
    sd.wait()
    status_var.set("Playback finished.")

# ==============================
# Menu Bar
# ==============================
menubar = Menu(root)
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="Open", command=sinput)
menubar.add_cascade(label="File", menu=file_menu)

record_menu = Menu(menubar, tearoff=0)
record_menu.add_command(label="Start Recording", command=start_recording)
record_menu.add_command(label="Stop Recording", command=stop_recording)
record_menu.add_command(label="Play Recording", command=play_recording)
record_menu.add_command(label="Save Recording", command=save_recording)
menubar.add_cascade(label="Record", menu=record_menu)

root.config(menu=menubar)
root.mainloop()
