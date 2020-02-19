from keras import Input, Model
from keras.backend import ctc_batch_cost, shape, reshape, permute_dimensions, cast
from keras.layers import Activation, Bidirectional, Reshape
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, Dropout
from keras.layers import LSTM
from keras.models import Sequential


def batch_output(batch, logs):
    print('\nFinished batch: ' + str(batch))
    print(logs)


def create_model(image_height, vocabulary_size):
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
    model = Bidirectional(LSTM(256, return_sequences=True))(model)
    model = Dropout(0.5)(model)
    model = Bidirectional(LSTM(256, return_sequences=True))(model)
    model = Dropout(0.5)(model)
    model = Dense(vocabulary_size + 1)(model)
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

    return model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def reshape_for_rnn(x, image_height, num_of_filters, input_shape, dim_reduction):
    x = permute_dimensions(x, (1, 0, 2, 3))
    feature_dim = num_of_filters * (image_height / dim_reduction)
    feature_dim = cast(feature_dim, "int32")
    feature_width = input_shape[1] / dim_reduction
    feature_width = cast(feature_width, "int32")
    new_x_shape = (cast(input_shape[0], "int32"), feature_width, feature_dim)
    return reshape(x, new_x_shape)
