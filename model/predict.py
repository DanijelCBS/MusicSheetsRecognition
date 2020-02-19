import os
from .utils import semantic_to_midi
import cv2
import numpy as np
from .model import create_model
from .utils import resize, normalize
from .tf_end_to_end.tf_predict import predict_tf


def predict_and_create_midi(slices, name, image_height, app):
    name = os.path.splitext(name)[0]
    predictions = []

    for i, slice in enumerate(slices):
        slice_pred = predict(slice, image_height, True, app)
        slice_pred_concat = ' '.join(slice_pred)
        slice_path = os.path.join(app.root_path, 'model\\data\\predictions', name + '__slice' + str(i) + '.semantic')
        with open(slice_path, 'w') as semantic_file:
            semantic_file.write(slice_pred_concat)
        predictions.append(slice_pred)

    semantic_concat = [j for i in predictions for j in i]
    semantic = ' '.join(semantic_concat)

    semantic_file_path = os.path.join(app.root_path, 'model\\data\\predictions', name + '.semantic')

    with open(semantic_file_path, 'w') as semantic_file:
        semantic_file.write(semantic)

    midi_file_path = os.path.join(app.root_path, 'model\\data\\predictions', name + '.mid')
    converter_path = os.path.join(app.root_path, 'model\\semantic_to_midi.jar')
    semantic_to_midi(semantic_file_path, midi_file_path, converter_path)

    return midi_file_path


def read_vocabulary(vocabulary_path):
    dict_file = open(vocabulary_path, 'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    return int2word


def predict(image, image_height, tf, app):
    voc_path = os.path.join(app.root_path, 'model\\data\\vocabulary_semantic.txt')
    int2word = read_vocabulary(voc_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not tf:
        model = create_model(image_height, len(int2word))
        model.load_weights('weights-13-0.0006.hdf5')
        image = resize(image, image_height)
        image = normalize(image)
        image = np.asarray(image).reshape(1, image.shape[1], image.shape[0], 1)

        seq_lengths = [image.shape[1] / 2 ** 4]

        labels_np = np.ones([8, 32])
        input_length = np.array(seq_lengths)
        label_length = np.ones([8, 1])

        inputs = {'the_input': image,
                  'the_labels': labels_np,
                  'input_length': input_length,
                  'label_length': label_length,
                  }

        prediction = model.predict(inputs)

        str_prediction = []
        for w in prediction[0]:
            str_prediction.append(int2word[int(w)])

        return str_prediction
    else:
        model_path = os.path.join(app.root_path, 'model\\semantic_model\\semantic_model.meta')
        return predict_tf(image, int2word, model_path)
