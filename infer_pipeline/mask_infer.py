import cv2
import numpy as np
 
#self.graph = tf.get_default_graph()
#self.model = model_from_checkpoint_path("checkpoints/resnet50_unet_1")
model = model_from_checkpoint_path("checkpoints/resnet50_unet_1")

cap = cv2.VideoCapture(0)
while True:
    # ret, frame = cap.read()
    ret, frame = self.cap_video.read()
    
    if ret:
        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgbImage = cv2.resize(rgbImage, (WINDOW_WIDTH, WINDOW_HEIGHT- 240))

        with self.graph.as_default():
            mask, mask_image = predict(
                model=self.model,
                inp=frame
            )

            mask_image = np.where(mask_image == class_colors[29], frame, mask_image)
