import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchvision import models, transforms as T

# === CONFIG
UPLOAD_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

class config:
    sampling_rate = 44100
    duration = 2
    samples = sampling_rate * duration
    hop_length = 347 * duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === TRANSFORMS
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# === MODEL
def load_pretrained_model(model_path="resnet18_fold.pth"):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model.to(device)

model = load_pretrained_model("resnet18_fold.pth")
model.eval()

# === UTILS
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_audio(file_path):
    y, sr = librosa.load(file_path, sr=config.sampling_rate)
    if len(y) >= config.samples:
        y = y[:config.samples]
    else:
        pad = config.samples - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    return y

def audio_to_mel(y):
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=config.sampling_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        fmin=config.fmin,
        fmax=config.fmax
    )
    return librosa.power_to_db(mels).astype(np.float32)

def save_mel_image(mels, path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mels, sr=config.sampling_rate,
                             hop_length=config.hop_length, fmin=config.fmin,
                             fmax=config.fmax, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def predict(mels):
    mels = (mels - mels.mean()) / mels.std()
    image = np.stack([mels] * 3, axis=-1)
    image = torch.tensor(image).permute(2, 0, 1).float()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        return output.item()

# === ROUTES
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.wav')
            file.save(file_path)

            y = read_audio(file_path)
            mels = audio_to_mel(y)

            spec_path = os.path.join(app.config['UPLOAD_FOLDER'], 'spectrogram.png')
            save_mel_image(mels, spec_path)

            pred_score = predict(mels)
            result = "Chainsaw Detected!!" if pred_score > 0.5 else "No Chainsaw!!"

            return render_template('index.html',
                                   image_path='output/spectrogram.png',
                                   audio_path='output/uploaded.wav',
                                   prediction=result)

    return render_template('index.html', image_path=None, prediction=None)

# === Run
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
