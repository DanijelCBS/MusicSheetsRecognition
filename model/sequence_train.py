import os

import cv2
import numpy as np
from keras.utils import Sequence
import random

from model import ctc_utils


class SequenceFactory:
    def __init__(self, corpus_dirpath, corpus_filepath, dictionary_path, batch_size, img_height, img_channels,
                 distortions=False, val_split=0.0):
        self.corpus_dirpath = corpus_dirpath
        self.batch_size = batch_size
        self.distortions = distortions
        self.img_height = img_height
        self.img_channels = img_channels
        self.corpus_filepath = corpus_filepath

        # Corpus
        corpus_file = open(corpus_filepath, 'r')
        corpus_list = corpus_file.read().splitlines()
        corpus_file.close()

        # Dictionary
        self.word2int = {}
        self.int2word = {}

        dict_file = open(dictionary_path, 'r')
        dict_list = dict_file.read().splitlines()
        for word in dict_list:
            word_idx = len(self.word2int)
            self.word2int[word] = word_idx
            self.int2word[word_idx] = word
        dict_file.close()

        self.vocabulary_size = len(self.word2int)

        random.shuffle(corpus_list)
        val_idx = int(len(corpus_list) * val_split)
        self.training_list = corpus_list[val_idx:]
        self.validation_list = corpus_list[:val_idx]

        print('Training set size: %d', str(len(self.training_list)))
        print('Validation set size: %d', str(len(self.validation_list)))

    def get_training_sequence(self):
        train = SequenceTrain(self.training_list, self.corpus_dirpath, self.corpus_filepath, self.batch_size,
                              self.distortions, self.img_height, self.img_channels, self.word2int, self.int2word)
        return train

    def get_validation_sequence(self):
        validation = SequenceTrain(self.validation_list, self.corpus_dirpath, self.corpus_filepath, self.batch_size,
                                   self.distortions, self.img_height, self.img_channels, self.word2int, self.int2word)
        return validation


class SequenceTrain(Sequence):
    PAD_COLUMN = 0

    def __init__(self, data_list, corpus_dirpath, corpus_filepath, batch_size, distortions, img_height, img_channels,
                 word2int, int2word):
        self.data_list = data_list
        self.corpus_dirpath = corpus_dirpath
        self.corpus_filepath = corpus_filepath
        self.batch_size = batch_size
        self.distortions = distortions
        self.img_height = img_height
        self.word2int = word2int
        self.int2word = int2word
        self.img_channels = img_channels

    def __getitem__(self, index):
        images = []
        labels = []
        current_idx = index * self.batch_size

        for i in range(self.batch_size):
            sample_filepath = self.data_list[(current_idx + i) % len(self.data_list)]
            sample_fullpath = os.path.join(self.corpus_dirpath, sample_filepath, sample_filepath)

            if self.distortions:
                sample_img = cv2.imread(sample_fullpath + '_distorted.jpg', False)
            else:
                sample_img = cv2.imread(sample_fullpath + '.png', False)

            sample_img = ctc_utils.resize(sample_img, self.img_height)
            images.append(ctc_utils.normalize(sample_img))

            sample_full_filepath = sample_fullpath + '.semantic'

            sample_gt_file = open(sample_full_filepath, 'r')
            sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
            sample_gt_file.close()

            labels.append([self.word2int[lab] for lab in sample_gt_plain])

        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)

        self.batch_images = np.ones(shape=[self.batch_size,
                                           self.img_height,
                                           max_image_width,
                                           self.img_channels], dtype=np.float32) * self.PAD_COLUMN

        for i, img in enumerate(images):
            self.batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

        return np.array([self.batch_images, np.array(labels)])

    def __len__(self):
        return int(np.ceil(len(self.data_list) / float(self.batch_size)))
