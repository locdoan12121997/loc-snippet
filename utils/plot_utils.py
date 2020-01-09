
import numpy as np
import cv2
import matplotlib

# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras


def draw_boxes(image, boxes, labels, obj_thresh,  quiet=True):
    nb_box =0
    face_center = [0,0]
    for box in boxes:
        
        # elinminate face smaller than 30X30
       # if (box.xmax-box.xmin) < 50 and (box.ymax-box.ymin) < 50:
       #     continue
        
        # draw the box
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
        # judge the box is a face or not----ZCY
        if label >= 0:
            nb_box+=1
            
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1] # this is the text size, a small rectangle
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  
            
    
            face_center=[(box.xmin+box.xmax)/2,(box.ymin+box.ymax)/2]
            #face=image[max(box.ymin,0):min(box.ymax,image.shape[1]), max(0,box.xmin):min(box.xmax,image.shape[0]), :]
            #img_name=(image_path.split('/')[-1]).split('.')[-2]
            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.imwrite('D:/CV/mypaper/test3' +'_%s' % nb_box + '.jpg', image)
# =============================================================================
#             cv2.fillPoly(img=image, pts=[region], color=get_color(label))
#             cv2.putText(img=image, 
#                         text=label_str, 
#                         org=(box.xmin+13, box.ymin - 13), 
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                         fontScale=1e-3 * image.shape[0], 
#                         color=(0,0,0), 
#                         thickness=2)
# =============================================================================
            
    return image, nb_box, face_center  


def get_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    code originally from https://github.com/fizyr/keras-retinanet/
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
    """
    if label < len(colors):
        return colors[label]
    else:
        print('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)

colors = [
    [31  , 0   , 255] ,
    [0   , 159 , 255] ,
    [255 , 95  , 0]   ,
    [255 , 19  , 0]   ,
    [255 , 0   , 0]   ,
    [255 , 38  , 0]   ,
    [0   , 255 , 25]  ,
    [255 , 0   , 133] ,
    [255 , 172 , 0]   ,
    [108 , 0   , 255] ,
    [0   , 82  , 255] ,
    [0   , 255 , 6]   ,
    [255 , 0   , 152] ,
    [223 , 0   , 255] ,
    [12  , 0   , 255] ,
    [0   , 255 , 178] ,
    [108 , 255 , 0]   ,
    [184 , 0   , 255] ,
    [255 , 0   , 76]  ,
    [146 , 255 , 0]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 248 , 0]   ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    [89  , 255 , 0]   ,
    [127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 255] ,
    [0   , 255 , 216] ,
    [0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    [31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    [0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    [0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    [12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    [0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    [0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    [70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    [50  , 255 , 0]
]

# def get_color_table(class_num, seed=2):
#     np.random.seed(seed)
#     color_table = {}
#     for i in range(class_num):
#         color_table[i] = [np.random.randint(0, 255) for _ in range(3)]
#     return color_table

class TrainingPlot(keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        # Initialize the lists for holding the logs, losses and accuracies
        self.save_dir = save_dir
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        pass

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available)  # to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("Training Accuracy and Loss")
            ax1.plot(N, self.acc, color='red', label="train_acc")
            ax1.plot(N, self.val_acc, color='green', label="val_acc")
            ax1.set(ylabel="Accuracy")
            ax1.legend()

            ax2.plot(N, self.losses, color='red', label="train_loss")
            ax2.plot(N, self.val_losses, color='green', label="val_loss")
            ax2.set(ylabel="Loss", xlabel="Epoch")
            ax2.legend()
            fig.savefig('{}/log.png'.format(self.save_dir))
            plt.close()
            # plt.figure()
            # plt.plot(N, self.losses, label="train_loss")
            # plt.plot(N, self.acc, label="train_acc")
            # plt.plot(N, self.val_losses, label="val_loss")
            # plt.plot(N, self.val_acc, label="val_acc")
            # plt.title("Training Loss and Accuracy")
            # plt.xlabel("Epoch #")
            # plt.ylabel("Loss/Accuracy")
            # plt.legend()
            # plt.show()
            # # Make sure there exists a folder called output in the current directory
            # # or replace 'output' with whatever directory you want to put in the plots
            # plt.savefig('{}/log.png'.format(self.save_dir))
            # plt.close()
