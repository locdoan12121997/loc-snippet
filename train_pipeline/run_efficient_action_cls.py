"""
aioz.aiar.truongle - Sep 24, 2019
run action cls
0-use phone
1-drink
2-eat
3-cover face
4-talking
5-sleep
6-record pen
7-record phone
"""
import cv2
import os
import argparse
import numpy as np
from helpers import helpers
from helpers.action_wrapper import ActionEstimate
from helpers.yolo3_detection_wrapper import YoloDetection

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parse = argparse.ArgumentParser()
parse.add_argument('--method', default='effB1_fcl_affineScale', type=str, help='overview method use for training')
parse.add_argument('--video_in', default="data/demo/input/action_many_test_1.mp4", type=str, help='video input')
parse.add_argument('--out_dir', default="data/demo/output", type=str, help='directory store video output')
parse.add_argument('--model_pth', default='saved_models/effB1_fcl_affineScale_test/myData_effB1_fcl_affineScale_model.025.h5')
arg = parse.parse_args()


def take_crop(boxes, input_img, ratio=0):
    persons = []
    img_h, img_w, _ = np.shape(input_img)

    if len(boxes) > 0:
        for i, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb
            w = xmax - xmin
            h = ymax - ymin
            xw1 = max(int(xmin - ratio * w), 0)
            yw1 = max(int(ymin - ratio * h), 0)
            xw2 = min(int(xmax + ratio * w), img_w - 1)
            yw2 = min(int(ymax + ratio * h), img_h - 1)
            crop = input_img[yw1:yw2, xw1:xw2]
            persons.append(crop)
    return persons


def main():
    # init
    helpers.check_dir(arg.out_dir)
    detector = YoloDetection(graph_path="graphs/yolov3_object_detection.pb")
    act_estimator = ActionEstimate(arg.model_pth)
    cap = cv2.VideoCapture(arg.video_in)

    # get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # init writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = os.path.join(arg.out_dir, arg.method + "_" + os.path.split(arg.video_in)[-1].split('.')[0] + ".mp4")
    writer = cv2.VideoWriter("data/demo/output/action_upstair.mp4", fourcc, fps, (int(vid_w), int(vid_h)))

    # process
    count = 0
    while cap.isOpened() and count < total_frame:
        ret, frame = cap.read()
        if ret:
            person_box = detector.predict(frame)
            person = take_crop(person_box, frame)
            if len(person) > 0:
                act_s = act_estimator.predict(person)
                frame = helpers.draw_act(frame, person_box, act_s)
            # write
            writer.write(frame)
            frame_show = cv2.resize(frame, (int(vid_w/2), int(vid_h/2)))
            cv2.imshow("task", frame_show)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
