"""
aioz.aiar.truongle - Oct 01, 2019
crop faces - build data
"""
import cv2
import os
import argparse
import numpy as np
from config import Config
from src.emotion_estimation.helpers import helpers
from inference_models.face_detection_inference_yolov3 import FaceDetection

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parse = argparse.ArgumentParser()
parse.add_argument('--video_in', default="backup_tutorial/age_gender.mp4", help="input video for crop faces")
parse.add_argument('--out_dir', default="backup_tutorial/data/fine_tune", help="directory store crop images")
arg = parse.parse_args()


def take_faces(boxes, input_img, ratio=0.15):
    faces = []
    img_h, img_w, _ = np.shape(input_img)

    if len(boxes) > 0:
        for i, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb[:4]
            w = xmax - xmin
            h = ymax - ymin
            xw1 = max(int(xmin - ratio * w), 0)
            yw1 = max(int(ymin - ratio * h), 0)
            xw2 = min(int(xmax + ratio * w), img_w - 1)
            yw2 = min(int(ymax + ratio * h), img_h - 1)
            crop = input_img[yw1:yw2, xw1:xw2]
            faces.append(crop)
    return faces


def main():
    # init
    helpers.check_dir(arg.out_dir)
    config = Config()
    # FACE DETECTOR
    face_detection = FaceDetection(config)
    # get video information
    cap = cv2.VideoCapture(arg.video_in)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # process
    count = 0
    while cap.isOpened() and count < total_frame:
        ret, frame = cap.read()
        if ret:
            faces_box = face_detection.process_detection(frame)
            faces = take_faces(faces_box, frame)
            # WRITE CROP
            for idx, crop in enumerate(faces):
                name = helpers.take_name(name_df='crop_%.5d_' % count, index=idx, len_max=2)
                save_pth = os.path.join(arg.out_dir, name)
                cv2.imwrite(save_pth, crop)

            frame_show = cv2.resize(frame, (int(vid_w/2), int(vid_h/2)))
            cv2.imshow("task", frame_show)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
