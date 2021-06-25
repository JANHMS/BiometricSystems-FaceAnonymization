import gzip
import shutil

import dlib
import csv
import face_recognition.api
import numpy as np
from PIL import ImageFile

import face_recognition
import face_recognition_models

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

known_encodings = []
known_index = []

NUM_IMG = 249
GALLERY_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/reference'
TEST_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/results'


def _face_encodings_(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = face_recognition.api._raw_face_landmarks(face_image, known_face_locations, model="large")
    return [np.array(face_encoder.compute_face_descriptor(face_image, landmark_set, num_jitters)) for
            landmark_set in raw_landmarks]


def main():
    text_in = input(' Run all tests? [yes,no]')
    auto = str(text_in)
    if auto == 'yes':
        # type_list = [ 'naive' ]
        type_list = [ 'reverse' ]
        method_list = ['blur', 'pixelate', 'noise' ]

        for type_item in type_list:
            TYPE = type_item
            print(' Chosen type:', TYPE)
            for method_item in method_list:
                METHOD = method_item
                print(' Chosen method is', METHOD)
                if TYPE == 'naive':
                    naive(METHOD)
                elif TYPE == 'reverse':
                    reverse(METHOD)
    else:
        text_in = input(' Method to becnhmark [blur, pixelated, noise, DeepPrivacy]:')
        METHOD = str(text_in)
        if METHOD not in ['blur', 'pixelated', 'noise', 'DeepPrivacy']: raise Exception('Wrong Method')
        print(' Chosen method is', METHOD)
        text_in = input(' Method to becnhmark [naive, reverse ]:')
        TYPE = str(text_in)
        print(' Chosen type:', TYPE)

        if TYPE == 'naive':
            naive(METHOD)
        elif TYPE == 'reverse':
            reverse(METHOD)
        else:
            raise Exception('Wrong Type')

def reverse(method):
    output = list()
    range_ = [ 25,  35, 50, 90 ]

    for k in range_:
        load_gallery('reverse', NUM_IMG, method=method, k=k)
        t = stats('reverse', method, k)
        print('', t)
        output.append(t)

    path_ = TEST_PATH + '/stats/' + method + '_' + 'reverse' + '_' + 'result.csv'

    np.savetxt(path_, output, delimiter=',')


def naive(method):
    load_gallery('naive', NUM_IMG)
    output = list()
    range_ = [ 5 ]

    for k in range_:
        t = stats('naive', method, k)
        print(' ', t)
        output.append(t)

    path_ = TEST_PATH + '/stats/' + method + '_' + 'naive' + '_' + 'result.csv'

    np.savetxt(path_, output, delimiter=',')


def load_gallery(type_: 'str', num_img: 'int', method='none', k=0):
    known_encodings.clear()
    known_index.clear()
    path = ''
    if type_ == 'naive':
        path = GALLERY_PATH
    elif type_ == 'reverse':
        path = TEST_PATH + '/' + method + '/' + str(k)
        if k == 0: raise Exception(' Wrong Reference')
    else:
        raise Exception(' Wrong Type')

    for index in range(1, num_img):
        image_path = path + '/' + str(index) + '.jpg'
        img = face_recognition.load_image_file(image_path)
        encodings = _face_encodings_(img, num_jitters=1)
        if len(encodings) == 0:
            # print("WARNING: No faces found in {}. Ignoring file.".format(index))
            pass
        elif len(encodings) > 1:
            # print("WARNING: More than one face found in {}. Only considering the first face.".format(index))
            known_encodings.append(encodings[0])
            known_index.append(index)
        else:
            known_encodings.append(encodings[0])
            known_index.append(index)

    print(' Reference Images are processed:', len(known_encodings), 'faces in the gallery')


def stats(type_, method, k):
    if type_ == 'naive':
        PATH = TEST_PATH + '/' + method + '/' + str(k)
    elif type_ == 'reverse':
        PATH = GALLERY_PATH

    failure_to_acquire = 0
    false_match_rate = 0
    false_non_match_rate = 0
    true_positive = 0
    genuine_score = list()
    non_mated_score = list()

    if len(known_encodings) == 0:
        while len(genuine_score) < NUM_IMG:
            genuine_score.append(0.0)
        while len(non_mated_score) < (NUM_IMG * (NUM_IMG - 1)):
            non_mated_score.append(0.0)

        non_mated_score = np.asarray(non_mated_score)
        genuine_score = np.asarray(genuine_score)

        pathmated = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + 'mated_score.txt'
        path_non_mated = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_non_mated_score.txt'
        np.savetxt(pathmated, genuine_score)
        np.savetxt(path_non_mated, non_mated_score)

        with open(pathmated, 'rb') as f_in:
            with gzip.open(pathmated + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        with open(path_non_mated, 'rb') as f_in:
            with gzip.open(path_non_mated + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return [int(k), 1.0, 0.0, 1.0, 0.0]

    for i in range(1, len(known_encodings)):
        # FACE ENCODING
        unknown_image = face_recognition.load_image_file(PATH + '/' + str(i) + '.jpg')
        unknown_face_encoding = _face_encodings_(unknown_image)
        if len(unknown_face_encoding) == 0:
            failure_to_acquire += 1
            false_non_match_rate += 1
        else:
            unknown_face_encoding = unknown_face_encoding[0]
            results = face_recognition.compare_faces(known_encodings, unknown_face_encoding)
            ## GENUINE AND non_mated SCORES
            distance = face_recognition.api.face_distance(known_encodings, unknown_face_encoding)
            similarity = 1 / (distance + 1)

            if not True in results: failure_to_acquire += 1

            false_match_rate += sum(results)

            index = -1
            for el in known_index:
                if el == i:
                    index = el
            # print(index, i)

            index = int(index) - 1
            
            # setting index to -2 to match correct files
            if index == -2:
                false_non_match_rate += 1
                genuine_score.append(0.0)
            elif not results[index]:
                false_non_match_rate += 1
                genuine_score.append(similarity[index])
            elif results[index]:
                genuine_score.append(similarity[index])
                false_match_rate -= 1
                true_positive += 1

            if len(distance) > 0:
                for j in range(0, len(distance)):
                    if not known_index[j] == i:
                        non_mated_score.append(similarity[j])

    ## Account for unidentified faces
    while len(genuine_score) < NUM_IMG:
        genuine_score.append(0.0)
    while len(non_mated_score) < (NUM_IMG * (NUM_IMG - 1)):
        non_mated_score.append(0.0)

    non_mated_score = np.asarray(non_mated_score)
    genuine_score = np.asarray(genuine_score)

    pathmated = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + 'mated_score.txt'
    path_non_mated = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_non_mated_score.txt'
    np.savetxt(pathmated, genuine_score)
    np.savetxt(path_non_mated, non_mated_score)

    with open(pathmated, 'rb') as f_in:
        with gzip.open(pathmated + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(path_non_mated, 'rb') as f_in:
        with gzip.open(path_non_mated + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not type_ == 'naive':
        failure_to_acquire += NUM_IMG - len(known_encodings)
        false_non_match_rate += NUM_IMG - len(known_encodings)

    failure_to_acquire /= NUM_IMG
    false_non_match_rate /= NUM_IMG
    false_match_rate /= NUM_IMG * (NUM_IMG - 1)

    return [int(k), failure_to_acquire, false_match_rate, false_non_match_rate, true_positive]


if __name__ == '__main__':
    main()