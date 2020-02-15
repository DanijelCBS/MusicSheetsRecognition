from sequence_train import SequenceFactory
from keras.callbacks import ModelCheckpoint
from model import *

def train_network(model, train_generator, val_generator):
    weights_path = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit_generator(generator, epochs=256, callbacks=callbacks_list, validation_data=val_generator, 
    	validation_freq=4, workers=2, use_multiprocessing=True)


if __name__ == '__main__':
	sequence_factory = SequenceFactory('./data/primus_dataset', './data/train.txt', 
		'./data/vocabulary_semantic.txt', 16, 128, 1, False, 0.1)
	model = create_model(4, sequence_factory.vocabulary_size)
	train_sequence = sequence_factory.get_training_sequence()
	val_sequence = sequence_factory.get_validation_sequence()
	train_network(model, train_sequence, val_sequence)