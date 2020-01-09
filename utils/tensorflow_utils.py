"""
aioz.aiar.truongle
crop, resize with tensorflow
"""
import tensorflow as tf
import numpy as np


def load_tf_graph(graph_path):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


class ImageCropper:
    def __init__(self):
        # crop images by bboxes
        self.tf_image = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.tf_bboxes = tf.placeholder(tf.float32, shape=(None, 4))
        self.tf_box_ind = tf.placeholder(tf.int32, shape=(None))
        self.tf_crop_size = tf.placeholder(tf.int32, shape=(2))

        self.tf_cropped_images = tf.image.crop_and_resize(self.tf_image,
                                                          self.tf_bboxes,
                                                          box_ind=self.tf_box_ind,
                                                          crop_size=self.tf_crop_size)
        self.crop_sess = tf.Session()
    
    def tf_run_crop(self, image, bboxes, crop_size=(64, 64), margin=(0.2, 0.2, 0.2, 0.2)):
        extended_bboxes = bboxes[:, :4].copy()
        widths, heights = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]

        extended_bboxes[:, 0] -= margin[0] * widths
        extended_bboxes[:, 1] -= margin[1] * heights
        extended_bboxes[:, 2] += margin[2] * widths
        extended_bboxes[:, 3] += margin[3] * heights
        extended_bboxes = np.clip(extended_bboxes, a_min=0, a_max=image.shape[1])
        extended_bboxes[:, 3] = np.clip(extended_bboxes[:, 3], a_min=0, a_max=image.shape[0])

        # print(extended_bboxes)

        normalized_bboxes = extended_bboxes.copy()
        normalized_bboxes[:, 0] = extended_bboxes[:, 1] / image.shape[0]
        normalized_bboxes[:, 1] = extended_bboxes[:, 0] / image.shape[1]
        normalized_bboxes[:, 2] = extended_bboxes[:, 3] / image.shape[0]
        normalized_bboxes[:, 3] = extended_bboxes[:, 2] / image.shape[1]

        images = self.crop_sess.run(self.tf_cropped_images,
                                    feed_dict={
                                        self.tf_image: np.expand_dims(image, 0),
                                        self.tf_bboxes: normalized_bboxes,
                                        self.tf_box_ind: np.zeros(len(bboxes), dtype=int),
                                        self.tf_crop_size: crop_size
                                    })
        return images


class NMS:
    def __init__(self):
        self.tf_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        self.tf_scores = tf.placeholder(tf.float32, shape=(None, ))
        self.tf_max_output_size = tf.placeholder(tf.int32)
        self.tf_iou_threshold = tf.placeholder(tf.float32)
        self.tf_refind_idx = tf.image.non_max_suppression(self.tf_boxes,
                                                          self.tf_scores,
                                                          max_output_size=self.tf_max_output_size,
                                                          iou_threshold=self.tf_iou_threshold)
        self.nms_sess = tf.Session()

    # def do_nms(self, boxes, nms_thresh):
    #     if len(boxes) > 0:
    #         nb_class = len(boxes[0][5])
    #     else:
    #         return
            
    #     for c in range(nb_class):
    #         sorted_indices = np.argsort([-box[5][c] for box in boxes])

    #         for i in range(len(sorted_indices)):
    #             index_i = sorted_indices[i]

    #             if boxes[index_i][5][c] == 0: continue

    #             for j in range(i+1, len(sorted_indices)):
    #                 index_j = sorted_indices[j]

    #                 if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
    #                     boxes[index_j][5][c] = 0
    #     return boxes

    def do_nms(self, bboxes, scores, nms_thresh):
        refind_idx = self.nms_sess.run(self.tf_refind_idx,
                                       feed_dict={
                                            self.tf_boxes: bboxes,
                                            self.tf_scores: scores,
                                            self.tf_max_output_size: bboxes.shape[0],
                                            self.tf_iou_threshold: nms_thresh})
        refined_bboxes = bboxes[refind_idx]
        refined_bboxes[refined_bboxes < 0] = 0
        return refined_bboxes


class ImageResize:
    def __init__(self):
        # resize images
        self.tf_image = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.tf_size = tf.placeholder(tf.int32, shape=2)

        self.tf_resize_images = tf.image.resize_images(self.tf_image,
                                                       size=self.tf_size)
        self.resize_sess = tf.Session()

    def tf_run_resize(self, images, size):
        """images.shape : None, None, None, 3"""
        images = self.resize_sess.run(self.tf_resize_images,
                                      feed_dict={
                                          self.tf_image: images,
                                          self.tf_size: size
                                      })
        return images


class WhitenResize:
    def __init__(self):
        self.tf_image = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.tf_size = tf.placeholder(tf.int32, shape=2)
        tf_whiten_image = tf.image.per_image_standardization(self.tf_image)
        self.tf_whitenResize_images = tf.image.resize_images(tf_whiten_image,
                                                             size=self.tf_size)
        self.whiten_resize_sess = tf.Session()

    def tf_run_whiten_resize(self, images, size):
        images = self.whiten_resize_sess.run(self.tf_whitenResize_images,
                                             feed_dict={self.tf_image: images,
                                                        self.tf_size: size})
        return images


# TEST CLASS
if __name__ == '__main__':
    import numpy as np
    ims = np.random.randint(255, size=(2, 5, 5, 3))
    im_resize = WhitenResize()
    ims = im_resize.tf_run_whiten_resize(ims, (5, 5))
    print(ims)
    print(ims.shape)


