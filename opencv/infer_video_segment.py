import cv2
from keras_segmentation.predict import predict, model_from_checkpoint_path
import random


#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
model = model_from_checkpoint_path("checkpoints/resnet50_unet_1")

cap = cv2.VideoCapture("test.mp4")

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        mask, mask_image = predict(
            model=model,
            inp=frame
        )
        cv2.imshow('Frame', mask_image)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()

cv2.destroyAllWindows()
