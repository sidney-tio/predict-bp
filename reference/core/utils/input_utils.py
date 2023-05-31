import glob
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from settings import config

# import mediapipe as mp


class ImageReadError(Exception):
    pass


class InvalidArgumentError(Exception):
    pass


class FrameIterator:
    def __init__(self, input):
        self.input = input
        self.total_frames = 0
        if os.path.isfile(input):
            self.input_type = 'video'
            self.cap = cv2.VideoCapture(input)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif os.path.isdir(input):
            self.input_type = 'images'
            self.frame_paths = sorted(glob.glob(os.path.join(input, '**/*')))
            self.total_frames = len(self.frame_paths)
        else:
            raise ValueError('FrameIterator does not support this input type')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.total_frames:
            if self.input_type == 'video':
                success, image = self.cap.read()
                self.index += 1
                return success, image
            elif self.input_type == 'images':
                try:
                    success = True
                    image = cv2.imread(self.frame_paths[self.index])
                    self.index += 1
                except ImageReadError:
                    success = False
                return success, image
        else:
            raise StopIteration


def decode_face_to_rbg_avg(face: np.ndarray, type: str = 'red') -> float:
    """_summary_

    Parameters
    ----------
    face : np.ndarray
        _description_
    type : str, optional
        _description_, by default 'red'

    Returns
    -------
    float
        _description_
    """
    if type in ('red', 'r'):
        return np.sum(face[:, :, 0]) / face.size
    elif type in ('green', 'g'):
        return np.sum(face[:, :, 1]) / face.size
    elif type in ('blue', 'b'):
        return np.sum(face[:, :, 2]) / face.size
    else:
        raise TypeError(f'Unsupported type: {type}')


def preprocess_video(video_content, filename, **kwargs):
    # For short
    # mp_face_detection = mp.solutions.face_detection

    ext = os.path.splitext(filename)[1]
    tmp_video_file = tempfile.NamedTemporaryFile(
        prefix='bonfire_video_').name + filename
    with open(tmp_video_file, 'wb') as f:
        f.write(video_content)

    if ext == '.zip':
        import zipfile
        frames_dir = tempfile.NamedTemporaryFile(
            prefix='bonfire_video_').name + Path(filename).stem
        with zipfile.ZipFile(tmp_video_file, 'r') as f:
            f.extractall(frames_dir)

        frames_iter = FrameIterator(frames_dir)
        fps = kwargs.get('fps')
        if fps is None:
            raise InvalidArgumentError(
                'FPS must be provided when the input type is images')
    elif ext in ('.mp4', '.avi'):
        frames_iter = FrameIterator(tmp_video_file)
        fps = frames_iter.cap.get(cv2.CAP_PROP_FPS)
    else:
        raise ValueError(f'Does not support this input type: {ext}')

    # set up
    # cap = cv2.VideoCapture(tmp_video_file)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    input_dim = config.model.args.input_shape[0]
    frame_depth = config.model.args.frame_depth
    Xsub = np.zeros((frames_iter.total_frames, input_dim,
                    input_dim, 3), dtype=np.float32)
    # calculator = VitalSignsCalculator(sampling_rate=fps)
    inputs = {
        'dXsub': np.empty(0),
        'fps': 0,
        'red_avg_list': [],
        'green_avg_list': [],
        'blue_avg_list': [],
    }
    # with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # while cap.isOpened():
    #     success, image = cap.read()
    #     if not success:
    #         break
    for success, image in frames_iter:

        # image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        face = cv2.resize(image, (input_dim, input_dim),
                          interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Take average of green, blue and red of the chin regions
        red_avg = decode_face_to_rbg_avg(face, type='r')
        green_avg = decode_face_to_rbg_avg(face, type='g')
        blue_avg = decode_face_to_rbg_avg(face, type='b')
        # calculator.update(red_avg, green_avg, blue_avg)
        inputs['red_avg_list'].append(red_avg)
        inputs['green_avg_list'].append(green_avg)
        inputs['blue_avg_list'].append(blue_avg)

        # results = face_detection.process(image)

        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # if results.detections:
        #     # Take the first detection only
        #     detection = results.detections[0]
        #     rel_bbox = detection.location_data.relative_bounding_box
        #     x1, y1, w, h = rel_bbox.xmin, rel_bbox.ymin, rel_bbox.width, rel_bbox.height
        #     x1, y1 = int(image_width * x1), int(image_height * y1)
        #     w, h = int(image_width * w), int(image_height * h)

        #     # Crop face and resize it
        #     # TODO: maybe we need to remove this step in the future?
        #     face = image[y1:y1+h, x1:x1+w, :]
        #     face = cv2.resize(face, (input_dim, input_dim))
        #     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        #     face = face.astype(np.float32)

        #     # Take average of green, blue and red of the chin regions
        #     red_avg = decode_face_to_rbg_avg(face, type='r')
        #     green_avg = decode_face_to_rbg_avg(face, type='g')
        #     blue_avg = decode_face_to_rbg_avg(face, type='b')
        #     # calculator.update(red_avg, green_avg, blue_avg)
        #     inputs['red_avg_list'].append(red_avg)
        #     inputs['green_avg_list'].append(green_avg)
        #     inputs['blue_avg_list'].append(blue_avg)
        # else:
        #     face = np.zeros((input_dim, input_dim, 3),
        #                     dtype=np.uint8).astype(np.float32)

        Xsub[frame_idx, :, :, :] = face / 255.0
        frame_idx += 1

    # Normalized frames in the motion branch
    normalized_len = frames_iter.total_frames - 1
    dXsub = np.zeros((normalized_len, input_dim,
                     input_dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / \
            (Xsub[j+1, :, :, :] + Xsub[j, :, :, :] + 1e-12)
    dXsub = dXsub / np.std(dXsub)

    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:frames_iter.total_frames-1, :, :, :]

    dXsub = np.concatenate((dXsub, Xsub), axis=3)

    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    # Remove the temp file
    if os.path.isfile(tmp_video_file):
        os.remove(tmp_video_file)
    if ext == '.zip' and os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)

    inputs['dXsub'] = dXsub
    inputs['fps'] = fps
    # return dXsub, fps
    return inputs
