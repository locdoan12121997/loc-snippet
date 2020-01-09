"""
aioz.aiar.truongle - Sep 25, 2019
run age estimation with file h5
"""
import cv2
import time
import numpy as np
from config import Config
from src.face_aligner.face_aligner import FaceAligner
from inference_models.face_detection_inference_yolov3 import FaceDetection
from inference_models.age_estimation_mean_variance import AgeEstimatorH5
from data import Data_Demo
import utils.application_utils as utils

video_test = Data_Demo()
VIDEO_PATH = video_test.demo_incinema_input
HEAD_POSE_THRESH = 45


def main():
    config = Config()
    # face detector
    face_detection = FaceDetection(config)
    # face aligner
    face_aligner = FaceAligner(config, landmark_points='68')

    # age detector
    age_detection = AgeEstimatorH5(config)

    cap = cv2.VideoCapture(VIDEO_PATH)
    vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    time_pro = []
    while cap.isOpened() and count <= total_frame:
        ret, frame = cap.read()
        if ret:
            # try:
            rescaled_frame = cv2.resize(frame, (int(vid_w / 4), int(vid_h / 4)))
            bboxes = face_detection.process_detection(rescaled_frame)
            if bboxes is not None:
                bboxes = bboxes * 4
                bboxes = bboxes[:, :4]
                bboxes_mask = ((bboxes[:, 3] - bboxes[:, 1]) > 50) * \
                              ((bboxes[:, 2] - bboxes[:, 0]) > 50)
                bboxes = bboxes[bboxes_mask]
                if bboxes is not None:
                    faces = face_aligner.align(frame, bboxes)
                    # estimate age
                    faces_112_112 = utils.resize_batch(faces, (112, 112))
                    prev_time = time.time()
                    ages = age_detection.process_prediction(faces_112_112)
                    print("[INFO] Age: ", ages)
                    # showing fps
                    exec_time = time.time() - prev_time
                    time_pro.append(exec_time)
                    print("[INFO] Time: ", exec_time)
                    info = "FPS: " + str(round((1 / exec_time), 1))
                    cv2.putText(frame, text=info, org=(50, 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)
            frame = cv2.resize(frame, (int(vid_w / 2), int(vid_h / 2)))
            cv2.imshow("Demo", frame)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("[INFO] Mean time: ", np.mean(np.asarray(time_pro)))
    print("[INFO] FPS: ", total_frame / np.sum(np.asarray(time_pro)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
