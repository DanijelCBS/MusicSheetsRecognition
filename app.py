import numpy as np
from flask import Flask, render_template, request, flash, redirect

from model.utils import process_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(content_type):
    return content_type.split('/')[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No image')
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        flash('No selected image')
        return redirect(request.url)

    if file and allowed_file(file.content_type):
        nparr = np.fromstring(file.read(), np.uint8)
        result = process_image(nparr)  # input for neural network
        return 'OK'


if __name__ == '__main__':
    app.run(debug=True)
