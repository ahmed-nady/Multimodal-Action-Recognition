import concurrent.futures
import time
import cv2
import numpy as np
import os
import pickle
from mmcv import load
#======PKU postprocessing======#
""" 
1- load PKU xsub skeleton pkl file (list of dicts)
2- get person bbox from its joints coordinates
3- crop it or just get minimum rectange that involves performer(s) during video
"""

ann_file_train = 'pku_xsub_train.pkl'
ann_file_val = 'pku_xsub_test.pkl'
annos_train = load(ann_file_train)
annos_val = load(ann_file_val)
annotations = annos_train+annos_val

PKU_dataset_dir ='/ActionRecognitionDatasets/PKUSpatialAlignment224'
target_size = (224, 224)

class MMCompact:
    def __init__(self, padding=0.25, threshold=10, hw_ratio=1, allow_imgpad=False):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = (hw_ratio,hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def get_box(self, keypoint, img_shape):
        # will return x1, y1, x2, y2
        h, w = img_shape
        # Make NaN zero
        keypoint[np.isnan(keypoint)] = 0.
        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
        return (min_x, min_y, max_x, max_y)

    def compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            #pad_r = int(pad_r/2)
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs
def processVid(vidName,min_bbox):
    t1= time.time()
    INPUT_VIDEO_PATH = os.path.join(dataset_dir, vidName)
    OUTPUT_VIDEO_PATH = os.path.join(PKU_dataset_dir, vidName)
    if not os.path.exists(PKU_dataset_dir):
        os.makedirs(PKU_dataset_dir)
    # Open the input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    x, y, x2, y2 = min_bbox
    x,y,x2,y2 = int(x/2),int(y/2),int(x2/2),int(y2/2)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, target_size)

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame= cv2.resize(frame, (960,540))
        # ==crop frm and then resize it to the target size
        frm = cv2.resize(frame[y:y2, x:x2], target_size)
        out.write(frm)

        # Release resources
    cap.release()
    out.release()
    # except Exception as e:
    #     print(f"exception {e} occured in {vidName}")


RGBPoseCompact = MMCompact()

if __name__=='__main__':
    dataset_dir = '/media/hd2/ActionData/PKU-MMD/RGB_VIDEO_ACTIONS'

    multi_threading = True
    dataset_videos = []
    for vid_name in os.listdir(dataset_dir):
        dataset_videos.append(vid_name)
    print("num of videos, ",len(dataset_videos))
    if not multi_threading:
        for i in range(len(dataset_videos)):
            anno = [x for x in annotations if x['frame_dir'] == dataset_videos[i][:-4]]
            if len(anno)==0:
                continue
            min_bbox = RGBPoseCompact.get_box(anno[0]['keypoint'],anno[0]['img_shape'])
            processVid(dataset_videos[i],min_bbox)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            #executor.map(processVid, dataset_videos)
            for i in range(len(dataset_videos)):
                anno = [x for x in annotations if x['frame_dir'] == dataset_videos[i][:-4]]
                if len(anno) == 0:
                    print("missing : ",dataset_videos[i][:-4])
                    continue
                min_bbox = RGBPoseCompact.get_box(anno[0]['keypoint'],anno[0]['img_shape'])
                executor.submit(processVid,dataset_videos[i],min_bbox)
