from keras import Input, Model
from keras.backend import ctc_batch_cost, shape, reshape, permute_dimensions
from keras.layers import Activation
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, Dropout
from keras.layers import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Sequential


def create_model(num_of_conv_blocks, vocabulary_size):
    # convolutional part
    model = Sequential()
    input_data = Input(name='the_input', shape=(None, 128, 1), dtype='float32')
    input_shape = shape(input_data)
    num_of_filters = 32
    pool_size = 2
    for i in range(num_of_conv_blocks):
        model = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same')(input_data)
        num_of_filters *= 2
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(model)

    # cnn to rnn transition
    num_of_last_filters = num_of_filters * num_of_conv_blocks * 2
    dim_reduction = int(128 / pool_size ** num_of_conv_blocks)
    model = Lambda(lambda x: reshape_for_rnn(x, num_of_last_filters, input_shape, dim_reduction))(model)

    # reccurent part
    lstm_11 = LSTM(512, return_sequences=True)(model)
    lstm_12 = LSTM(512, return_sequences=True)(model)
    lstm1_merged = add([lstm_11, lstm_12])
    lstm_21 = LSTM(512, return_sequences=True)(lstm1_merged)
    lstm_22 = LSTM(512, return_sequences=True)(lstm1_merged)
    model = concatenate([lstm_21, lstm_22])
    model = Dropout(0.3)(model)
    model = Dense(vocabulary_size + 1)(model)
    y_pred_out = Activation('softmax', name='softmax')(model)
    Model(inputs=input_data, outputs=y_pred_out).summary()

    # ctc part
    labels = Input(name='the_labels', shape=[vocabulary_size + 1], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred_out, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred_out}, optimizer='adam')

    return model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def reshape_for_rnn(x, num_of_last_filters, input_shape, dim_reduction):
    x = permute_dimensions(x, (2, 0, 3, 1))
    feature_dim = num_of_last_filters * dim_reduction
    feature_width = input_shape[1] / dim_reduction
    new_x_shape = (feature_width, input_shape[0], feature_dim)
    return reshape(x, new_x_shape)


if __name__ == '__main__':
    model = create_model(4, 30)
