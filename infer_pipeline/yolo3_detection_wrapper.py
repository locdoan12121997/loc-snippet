"""
aioz.aiar.truongle - june 06, 2019
yolov3 object detection
Sub module is used for detecting person inside cinema
"""
import numpy as np
import tensorflow as tf
import cv2


class YoloDetection:
    def __init__(self, graph_path='yolov3_object_detection.pb', size=416, threshold=0.7, memory_fraction=0.3):
        # config gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False

        self.graph_fp = graph_path
        self.im_size = size
        self.threshold = threshold
        self.session = None

        self._load_graph()
        self._init_predictor()

    def _load_graph(self):
        print('[INFO] Load graph at {} ... '.format(self.graph_fp))
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # tf.get_default_graph().finalize()

    def _init_predictor(self):
        # print('[INFO] Init predictor ...')
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=self.config)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes_tensor = self.graph.get_tensor_by_name('concat_10:0')
            self.score_tensor = self.graph.get_tensor_by_name('concat_11:0')
            self.labels_tensor = self.graph.get_tensor_by_name('concat_12:0')

    def predict(self, image):

        height_ori, width_ori = image.shape[:2]
        img = cv2.resize(image, (self.im_size, self.im_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = self.session.run([self.boxes_tensor, self.score_tensor, self.labels_tensor],
                                                    feed_dict={self.image_tensor: img})

        # filter
        indices_ps = np.where(labels_ == 0)
        boxes_ps = boxes_[indices_ps]
        scores_ps = scores_[indices_ps]
        indices = np.where(scores_ps > self.threshold)
        boxes = boxes_ps[indices]

        # rescale the coordinates to the original image
        boxes[:, 0] *= (width_ori / float(self.im_size))
        boxes[:, 2] *= (width_ori / float(self.im_size))
        boxes[:, 1] *= (height_ori / float(self.im_size))
        boxes[:, 3] *= (height_ori / float(self.im_size))

        # check boxes
        idi = np.where(boxes[:, 0] < 0)[0]
        boxes[idi, 0] = 0
        idi = np.where(boxes[:, 1] < 0)[0]
        boxes[idi, 1] = 0
        idi = np.where(boxes[:, 2] > width_ori)[0]
        boxes[idi, 2] = width_ori
        idi = np.where(boxes[:, 3] > height_ori)[0]
        boxes[idi, 3] = height_ori

        # # filter boxes
        # print("boxes before: \n", boxes)
        # ratio_whs = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])
        # idi_ratio_l = np.where(ratio_whs > self.ratio_thr[0])[0]
        # # idi_ratio = idi_ratio_l
        # idi_ratio_h = np.where(ratio_whs < self.ratio_thr[1])[0]
        # idi_ratio = np.intersect1d(idi_ratio_l, idi_ratio_h)
        # boxes = boxes[idi_ratio]

        return boxes.astype(int)
