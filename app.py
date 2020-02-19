import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, session
import secrets
from model.predict import predict_and_create_midi
from model.utils import process_image, midi_to_musicxml

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html', musicXml=session['musicXml'])


def allowed_file(content_type):
    return content_type.split('/')[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No image', 'error')
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        flash('No selected image', 'error')
        return redirect(request.url)

    if file and allowed_file(file.content_type):
        nparr = np.fromstring(file.read(), np.uint8)
        result = process_image(nparr)  # input for neural network
        midi_path = predict_and_create_midi(result, file.filename, 32, app)  # midi file path
        musicXml = midi_to_musicxml(midi_path)
        session['musicXml'] = musicXml
        return redirect(url_for('result')), 201
    else:
        flash('Accepted formats are png, jpg and jpeg', 'error')
        return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
