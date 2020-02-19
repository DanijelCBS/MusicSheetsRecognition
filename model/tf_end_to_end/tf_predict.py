import tensorflow.compat.v1 as tf
from .ctc_utils import *
import numpy as np


def predict_tf(image, int2word, model_path):
    tf.disable_eager_execution()

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Restore weights
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, model_path[:-5])

    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.get_collection("logits")[0]

    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

    image = resize(image, HEIGHT)
    image = normalize(image)
    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)

    seq_lengths = [image.shape[2] / WIDTH_REDUCTION]

    prediction = sess.run(decoded,
                          feed_dict={
                              input: image,
                              seq_len: seq_lengths,
                              rnn_keep_prob: 1.0,
                          })

    str_predictions = sparse_tensor_to_strs(prediction)
    pred = []
    for w in str_predictions[0]:
        pred.append(int2word[w])

    return pred
