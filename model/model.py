from keras import Input, Model
from keras.backend import ctc_batch_cost, shape, reshape, permute_dimensions, cast
from keras.layers import Activation
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, Dropout
from keras.layers import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LambdaCallback
from sequence_train import SequenceFactory


def batch_output(batch, logs):
    print('Finished batch: ' + str(batch))
    print(logs)


def create_model(image_height, sequence_factory):
    # convolutional part
    model = Sequential()
    input_data = Input(name='the_input', shape=(None, image_height, 1), dtype='float32')
    input_shape = shape(input_data)
    num_of_filters = 32
    pool_size = 2
    dim_reduction = 1

    model = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same')(input_data)
    num_of_filters *= 2
    model = BatchNormalization()(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(model)
    dim_reduction *= 2

    model = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same')(model)
    num_of_filters *= 2
    model = BatchNormalization()(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(model)
    dim_reduction *= 2

    model = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same')(model)
    num_of_filters *= 2
    model = BatchNormalization()(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(model)
    dim_reduction *= 2

    model = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(model)
    dim_reduction *= 2

    # cnn to rnn transition
    model = Lambda(lambda x: reshape_for_rnn(x, image_height, num_of_filters, input_shape, dim_reduction))(model)

    # reccurent part
    lstm_11 = LSTM(512, return_sequences=True)(model)
    lstm_12 = LSTM(512, return_sequences=True)(model)
    lstm1_merged = add([lstm_11, lstm_12])
    lstm_21 = LSTM(512, return_sequences=True)(lstm1_merged)
    lstm_22 = LSTM(512, return_sequences=True)(lstm1_merged)
    model = concatenate([lstm_21, lstm_22])
    model = Dropout(0.3)(model)
    model = Dense(sequence_factory.vocabulary_size + 1)(model)
    y_pred_out = Activation('softmax', name='softmax')(model)
    Model(inputs=input_data, outputs=y_pred_out).summary()

    # ctc part
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='my_ctc')(
        [y_pred_out, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'my_ctc': lambda y_true, y_pred: y_pred_out}, optimizer='adam')

    weights_path = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    batch_log = LambdaCallback(on_batch_end=batch_output)
    callbacks_list = [checkpoint, batch_log]

    train_sequence = sequence_factory.get_training_sequence()
    val_sequence = sequence_factory.get_validation_sequence()

    model.fit_generator(train_sequence, epochs=256, callbacks=callbacks_list, validation_data=val_sequence,
                        validation_freq=4, workers=2, use_multiprocessing=True)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def reshape_for_rnn(x, image_height, num_of_filters, input_shape, dim_reduction):
    x = permute_dimensions(x, (1, 0, 2, 3))
    feature_dim = num_of_filters * (image_height / dim_reduction)
    feature_dim = cast(feature_dim, "int32")
    feature_width = input_shape[1] / dim_reduction
    feature_width = cast(feature_width, "int32")
    new_x_shape = (feature_width, cast(input_shape[0], "int32"), feature_dim)
    return reshape(x, new_x_shape)


if __name__ == '__main__':
    sequence_factory = SequenceFactory('data/primus_dataset', 'data/train.txt',
                                       'data/vocabulary_semantic.txt',
                                       1, 32, 1, False, 0.1)
    create_model(32, sequence_factory)
