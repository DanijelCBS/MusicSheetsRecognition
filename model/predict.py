import os
from .utils import semantic_to_midi


def predict_and_create_midi(slices, name):
    predictions = []
    for slice in slices:
        predictions.append(predict(slice))

    semantic = ' '.join(predictions)

    semantic_file_path = os.path.join('predictions', name + '.semantic')

    with open(semantic_file_path, 'w') as semantic_file:
        semantic_file.write(semantic)

    midi_file_path = os.path.join('predictions', name + '.mid')
    semantic_to_midi(semantic_file_path, midi_file_path)

    return midi_file_path


def predict(slice):
    return None
