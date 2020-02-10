import os

from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'OK'


if __name__ == '__main__':
    app.run(debug=True)
