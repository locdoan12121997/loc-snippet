from yolo3_detection_wrapper import YoloDetection
import cv2
import os
import glob

dir_name = "action_video/dark"

def crop_face_from_video(path_video, person_detector):
    name_folder = path_video.split('/')[2].split('.')[0]
    path_folder = "action_result/" + name_folder
    if os.path.isdir(path_folder) == False:
        os.mkdir(path_folder)

    cap = cv2.VideoCapture(path_video)
    counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            try:
                bounding_box = person_detector.predict(frame)[0]
                action_frame = frame[int(bounding_box[0] * 0.7): int(bounding_box[2] * 1.2), int(bounding_box[1]* 1.2) : int(bounding_box[3] * 1.2)]
                if len(action_frame):
                    cv2.imshow('frame', action_frame)
                else:
                    continue
                cv2.imwrite(path_folder + "/%d.jpg" % (counter), action_frame)
                counter += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
        else:
            break
    cap.release()


model = YoloDetection()
# files = [f for f in glob.glob(dir_name + "/*.mp4")]
files = ["action_video/dark/uyen.mp4", "action_video/dark/uyen2.mp4", "action_video/dark/uyen3.mp4"]
for f in files:
    print(f)
    crop_face_from_video(f, model)
