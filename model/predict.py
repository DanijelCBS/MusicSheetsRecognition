import cv2
import numpy as np
from model import create_model
from ctc_utils import resize, normalize


def read_vocabulary(vocabulary_path):
    dict_file = open(vocabulary_path, 'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    return int2word


def predict(model, weights_path, image_path, image_height):
    model.load_weights(weights_path)
    image = cv2.imread(image_path, False)
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

    print(prediction)
    for w in prediction[0]:
        print(int2word[int(w)]),
        print('\t')


if __name__ == '__main__':
    image_height = 32
    int2word = read_vocabulary('data/vocabulary_semantic.txt')
    model = create_model(image_height, len(int2word))
    predict(model, 'weights-13-0.0006.hdf5', 'data/000051652-1_2_1.png', image_height)
