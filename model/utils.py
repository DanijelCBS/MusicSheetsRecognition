import cv2
import numpy as np
import subprocess
import shlex


def process_image(nparr):
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_inv = 255 - img
    inv_gray = cv2.cvtColor(img_inv, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_dil = cv2.dilate(inv_gray, kernel, iterations=2)
    img_close = cv2.erode(img_dil, kernel, iterations=1)

    _, contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_lines = []
    for contour in contours:
        min_area_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(min_area_rect)
        box = np.reshape(box, [-1, ])
        x1 = (box[0], box[1])
        y1 = (box[2], box[3])
        x2 = (box[4], box[5])
        y2 = (box[6], box[7])
        width = abs(x2[0] - x1[0])
        height = abs(y1[1] - y2[1])

        if width > 400 and 50 <= height <= 120:
            (x, y, w, h) = cv2.boundingRect(contour)
            staff_lines.append(img[y:y + h, x:x + w])

    return np.array(staff_lines)


def semantic_to_midi(input_file_path, output_file_path):
    subprocess.call(shlex.split(
        f'java -cp semantic_to_midi.jar es.ua.dlsi.im3.omr.encoding.semantic.SemanticImporter {input_file_path} {output_file_path}'))


if __name__ == '__main__':
    semantic_to_midi('C:/Users/korisnik/Desktop/primus_conversor/000100973-1_1_2.semantic',
                     'C:/Users/korisnik/Desktop/primus_conversor/new.mid')
