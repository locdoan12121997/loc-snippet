class Config:
    def __init__(self):
        ### ROOT DIRECTORY ###
        self.root_dir = '/home/aioz-truong/MonData/Project/aiar_AICX_project'

        ### FACE DETECTION ###
        # self.prob_thresh = 0.5
        # self.nms_thresh = 0.1
        self.prob_thresh = 0.95
        self.nms_thresh = 0.3
        # Face Aligner
        self.aligner = 'models/aligner/shape_predictor_68_face_landmarks.dat'
        self.aligner_targets = 'models/aligner/targets_symm.txt'
        # Tiny Faces
        self.tiny_face_graph_path = 'models/detection/face/tiny_faces.pb'
        self.min_detections = 2
        # YoloV3
        self.yolov3_graph_path = 'models/face/2019_07_05_YoloV3_Face.pb'
        self.head_yolov3_graph_path = 'models/head/2019_07_05_YoloV3_Head.pb'
        self.net_h = 416
        self.net_w = 416
        self.anchors = [55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]

        ### HEAD DETECTION ###
        # Pyramid box
        self.pyramid_box_graph_path = 'models/head/2019_06_26_Truong_freeze_PyramidBox_VGG16_0.5_tiny_head_detection.pb'

        ### GENDER DETECTION ###
        # Xception
        self.xception_gender_graph_path = 'models/detection/gender/2019_06_13_pretrained_efficientB0_utk_fine_tune_afadgenderdetection_acc_98.245.h5'
        # self.xception_gender_graph_path = 'models/detection/gender/gender_real_data.h5'
        # EfficientNet
        self.efficient_net_gender_graph_path = 'models/gender/2019_06_13_pretrained_efficientB0_utk_fine_tune_afadgenderdetection_acc_98.245.h5'

        ### AGE ESTIMATOR ###
        # Mean Variance
        self.mean_variance_graph_path = 'models/age/2019_07_05_MeanVariance_Age.pb'
        # self.mean_variance_graph_path = 'models/detection/age/age_real_data.pb'

        ### EXPRESSION ###
        # ResNet + FER2013
        # self.expr_graph_path = 'models/detection/expression/resnet_FER.h5'
        self.expr_graph_path = 'models/expression/2019_07_05_EfficientNetB0_Expression.h5'
        # self.exprs = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.exprs = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Bored', 'Disgust', 'Fear']

        ### HEAD POSE ###
        # FSAnet
        self.fsanet_graph_path = 'models/headpose/2019_06_26_Truong_freeze_FSANet_headPosePrediction.pb'

        ### FACE ENCODER ###
        self.facenet_graph_path = 'models/tracking/face_features_extractor/2019_07_05_Facenet_FaceFeaturesExtractor.pb'
        # self.mobile_facenet_graph_path = 'models/tracking/face_features_extractor/mobile_facenet.pb'

        ### TRACKING ###
        self.nms_max_overlap = 1.0
        self.max_cosine_distance = 0.9
        self.max_euclidean_distance = 0.9
        self.nn_budget = 100
class Config:
    def __init__(self):
        ### ROOT DIRECTORY ###
        self.root_dir = '/home/aioz-chien/Project/aiar_camera_monitoring_project/camera_monitoring_combined_modules'

        ### FACE DETECTION ###
        # self.prob_thresh = 0.5
        # self.nms_thresh = 0.1
        self.prob_thresh = 0.95
        self.nms_thresh = 0.3
        # Face Aligner
        self.aligner = 'models/aligner/shape_predictor_68_face_landmarks.dat'
        self.aligner_targets = 'models/aligner/targets_symm.txt'
        # Tiny Faces
        self.tiny_face_graph_path = 'models/detection/face/tiny_faces.pb'
        self.min_detections = 2
        # YoloV3
        self.yolov3_graph_path = 'models/face/2019_07_05_YoloV3_Face.pb'
        self.head_yolov3_graph_path = 'models/head/2019_07_05_YoloV3_Head.pb'
        self.net_h = 416
        self.net_w = 416
        self.anchors = [55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]

        ### HEAD DETECTION ###
        # Pyramid box
        self.pyramid_box_graph_path = 'models/head/2019_06_26_Truong_freeze_PyramidBox_VGG16_0.5_tiny_head_detection.pb'

        ### GENDER DETECTION ###
        # Xception
        self.xception_gender_graph_path = 'models/detection/gender/2019_06_13_pretrained_efficientB0_utk_fine_tune_afadgenderdetection_acc_98.245.h5'
        # self.xception_gender_graph_path = 'models/detection/gender/gender_real_data.h5'
        # EfficientNet
        self.efficient_net_gender_graph_path = 'models/gender/2019_06_13_pretrained_efficientB0_utk_fine_tune_afadgenderdetection_acc_98.245.h5'

        ### AGE ESTIMATOR ###
        # Mean Variance
        self.mean_variance_graph_path = 'models/age/2019_07_05_MeanVariance_Age.pb'
        # self.mean_variance_graph_path = 'models/detection/age/age_real_data.pb'

        ### EXPRESSION ###
        # ResNet + FER2013
        # self.expr_graph_path = 'models/detection/expression/resnet_FER.h5'
        self.expr_graph_path = 'models/expression/2019_07_05_EfficientNetB0_Expression.h5'
        # self.exprs = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.exprs = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Bored', 'Disgust', 'Fear']

        ### HEAD POSE ###
        # FSAnet
        self.fsanet_graph_path = 'models/headpose/2019_06_26_Truong_freeze_FSANet_headPosePrediction.pb'

        ### FACE ENCODER ###
        self.facenet_graph_path = 'models/tracking/face_features_extractor/2019_07_05_Facenet_FaceFeaturesExtractor.pb'
        # self.mobile_facenet_graph_path = 'models/tracking/face_features_extractor/mobile_facenet.pb'

        ### TRACKING ###
        self.nms_max_overlap = 1.0
        self.max_cosine_distance = 0.9
        self.max_euclidean_distance = 0.9
        self.nn_budget = 100
