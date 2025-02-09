# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import pickle
import cv2

import numpy as np
from mmcv.fileio import FileClient
from scipy.stats import mode

from ..builder import PIPELINES
from .augmentations import Flip
import random
import warnings
from sklearn_extra.cluster import KMedoids
warnings.filterwarnings('ignore') # setting ignore as a parameter
@PIPELINES.register_module()
class MGSampleFrames(object):
    """Sample frames from the video.
    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".
    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='repeat_last',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):

        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx+1)
            except(ValueError):
                print(results['filename'])

        diff_score = results['skelton_motion_score']
        diff_score = np.power(diff_score, 0.5)
        sum_num = np.sum(diff_score)
        diff_score = diff_score / sum_num

        count = 0
        pic_diff = list()
        for i in range(len(diff_score)):
            count = count + diff_score[i]
            pic_diff.append(count)

        choose_index = list()

        if self.test_mode:
            choose_index.append(find_nearest(pic_diff, 1 / 32))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 1 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 2 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 3 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 4 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 5 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 6 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 7 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 8 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 9 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 10 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 11 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 12 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 13 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 14 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 15 / 16))



        else:
            choose_index.append(find_nearest(pic_diff, random.uniform(0, 1 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(1 / 16, 2 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(2 / 16, 3 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(3 / 16, 4 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(4 / 16, 5 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(5 / 16, 6 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(6 / 16, 7 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(7 / 16, 8 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(8 / 16, 9 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(9 / 16, 10 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(10 / 16, 11 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(11 / 16, 12 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(12 / 16, 13 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(13 / 16, 14 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(14 / 16, 15 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(15 / 16, 16 / 16)))
        #print(choose_index)
        #print("choose_index_MGS",choose_index)
        results['frame_inds'] = np.array(choose_index)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

        """
        total_frames = results['total_frames']
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')
        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results
        """

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

@PIPELINES.register_module()
class PoseSimGSampleFrames(object):
    """Sample frames from the video.
    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".
    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='repeat_last',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')


    def __call__(self, results):

        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx)
            except(ValueError):
                print(results['filename'])

        diff_score = results['pose_similarity_score']
        #diff_score = np.power(diff_score, 0.5)
        sum_num = np.sum(diff_score)
        diff_score = diff_score / sum_num

        count = 0
        pic_diff = list()
        for i in range(len(diff_score)):
            count = count + diff_score[i]
            pic_diff.append(count)

        choose_index = list()
        if self.test_mode:
            choose_index.append(find_nearest(pic_diff, 1 / 64))
            for i in range(1, 32):
                choose_index.append(find_nearest(pic_diff, 1 /64 + i / 32))
        else:
            choose_index.append(find_nearest(pic_diff, random.uniform(0, 1 / 32)))
            for i in range(1, 32):
                choose_index.append(find_nearest(pic_diff, random.uniform(i / 32, (i + 1) / 32)))
        #print(choose_index)
        #print("choose_index",choose_index)
        results['frame_inds'] = np.array(choose_index)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

@PIPELINES.register_module()
class ClusteringSampleFrames(object):
    """Sample frames from the video.
    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".
    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='repeat_last',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def normalize_pose(self,input):

        neck_point = (input[5, :] + input[6, :]) / 2
        scale = abs(input[5, :] - input[6, :])
        input_normalized = (input - neck_point) / (scale + 1)
        return input_normalized

    # do clustering

    def __call__(self, results):

        all_kps = results['keypoint']
        kp_shape = all_kps.shape
        num_frames = all_kps.shape[1]
        poses_coord_lst =[]
        for i in range(num_frames):

            wholeBPose_keypoints = all_kps[:, 0][0]
            temp = self.normalize_pose(wholeBPose_keypoints)
            x_coords = [pt[0] for pt in temp]
            y_coords = [pt[1] for pt in temp]
            poses_coord_lst.append(x_coords + y_coords)

        training_data = poses_coord_lst
        # clustering poses
        # Compute Kmedoids clustering
        cobj = KMedoids(n_clusters=16).fit(np.array(training_data))
        frame_inds = cobj.medoid_indices_
        choose_index =sorted(frame_inds)
        #print(sorted(frame_inds))
        #print("choose_index",choose_index)
        results['frame_inds'] = np.array(choose_index)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

@PIPELINES.register_module()
class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self, clip_len, num_clips=1, test_mode=False, seed=255):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """

        assert self.num_clips == 1
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        return inds

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """

        np.random.seed(self.seed)
        if num_frames < clip_len:
            # Then we use a simple strategy
            if num_frames < self.num_clips:
                start_inds = list(range(self.num_clips))
            else:
                start_inds = [
                    i * num_frames // self.num_clips
                    for i in range(self.num_clips)
                ]
            inds = np.concatenate(
                [np.arange(i, i + clip_len) for i in start_inds])
        elif clip_len <= num_frames < clip_len * 2:
            all_inds = []
            for i in range(self.num_clips):
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(self.num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds

    def __call__(self, results):
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index
        #print("frame_inds",len(inds),num_frames,inds)
        results['frame_inds'] = inds.astype(np.int)
        #print(f"results['frame_inds']: {results['frame_inds']}")
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str


@PIPELINES.register_module()
class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score"
    (optional), added or modified keys are "keypoint", "keypoint_score" (if
    applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        """Load keypoints given frame indices.

        Args:
            kp (np.ndarray): The keypoint coordinates.
            frame_inds (np.ndarray): The frame indices.
        """

        return [x[frame_inds].astype(np.float32) for x in kp]

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        """Load keypoint scores given frame indices.

        Args:
            kpscore (np.ndarray): The confidence scores of keypoints.
            frame_inds (np.ndarray): The frame indices.
        """

        return [x[frame_inds].astype(np.float32) for x in kpscore]

    def ntu_post_processing(self,results, padding=0.25):
        # ===get bbox that includes all these keypoints====#
        kp = results['keypoint']
        img_h, img_w = results['img_shape']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # # The compact area is too small
        # if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
        #     return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + padding)
        half_height = (max_y - min_y) / 2 * (1 + padding)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height
        if  np.isnan(min_x) or  np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y):
            print('Nan')
        min_x, min_y = round(min_x), round(min_y)
        max_x, max_y = round(max_x), round(max_y)

        # ==check image's boundaries
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(max_x, img_w), min(max_y, img_h)
        return (min_x, min_y, max_x, max_y)
    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            kpscore = results['keypoint_score']
            results['keypoint_score'] = kpscore[:,frame_inds].astype(np.float32)

        if 'keypoint' in results:
            results['keypoint'] = results['keypoint'][:, frame_inds].astype(
                np.float32)

            # #===set min_bbox ==
            # results['min_bbox'] =self.ntu_post_processing(results)
            # demonitor=1
            # real_img_shape = (1080//demonitor,1920//demonitor)
            # if real_img_shape != results['img_shape']:
            #     oh, ow = results['img_shape']
            #     nh, nw = real_img_shape
            #
            #     assert results['keypoint'].shape[-1] in [2, 3]
            #     results['keypoint'][..., 0] *= (nw / ow)
            #     results['keypoint'][..., 1] *= (nh / oh)
            #     #results['keypoint'] = results['keypoint'].astype(np.int32)
            #     results['img_shape'] = real_img_shape
            #     results['original_shape'] = real_img_shape

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class LoadKineticsPose:
    """Load Kinetics Pose given filename (The format should be pickle)

    Required keys are "filename", "total_frames", "img_shape", "frame_inds",
    "anno_inds" (for mmpose source, optional), added or modified keys are
    "keypoint", "keypoint_score".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        squeeze (bool): Whether to remove frames with no human pose.
            Default: True.
        max_person (int): The max number of persons in a frame. Default: 10.
        keypoint_weight (dict): The weight of keypoints. We set the confidence
            score of a person as the weighted sum of confidence scores of each
            joint. Persons with low confidence scores are dropped (if exceed
            max_person). Default: dict(face=1, torso=2, limb=3).
        source (str): The sources of the keypoints used. Choices are 'mmpose'
            and 'openpose-18'. Default: 'mmpose'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self,
                 io_backend='disk',
                 squeeze=True,
                 max_person=100,
                 keypoint_weight=dict(face=1, torso=2, limb=3),
                 source='mmpose',
                 **kwargs):

        self.io_backend = io_backend
        self.squeeze = squeeze
        self.max_person = max_person
        self.keypoint_weight = cp.deepcopy(keypoint_weight)
        self.source = source

        if source == 'openpose-18':
            self.kpsubset = dict(
                face=[0, 14, 15, 16, 17],
                torso=[1, 2, 8, 5, 11],
                limb=[3, 4, 6, 7, 9, 10, 12, 13])
        elif source == 'mmpose':
            self.kpsubset = dict(
                face=[0, 1, 2, 3, 4],
                torso=[5, 6, 11, 12],
                limb=[7, 8, 9, 10, 13, 14, 15, 16])
        else:
            raise NotImplementedError('Unknown source of Kinetics Pose')

        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):

        assert 'filename' in results
        filename = results.pop('filename')

        # only applicable to source == 'mmpose'
        anno_inds = None
        if 'anno_inds' in results:
            assert self.source == 'mmpose'
            anno_inds = results.pop('anno_inds')
        results.pop('box_score', None)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)

        # only the kp array is in the pickle file, each kp include x, y, score.
        kps = pickle.loads(bytes)

        total_frames = results['total_frames']

        frame_inds = results.pop('frame_inds')

        if anno_inds is not None:
            kps = kps[anno_inds]
            frame_inds = frame_inds[anno_inds]

        frame_inds = list(frame_inds)

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        # write it back
        results['total_frames'] = total_frames

        h, w = results['img_shape']
        if self.source == 'openpose-18':
            kps[:, :, 0] *= w
            kps[:, :, 1] *= h

        num_kp = kps.shape[1]
        num_person = mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_kp, 2],
                          dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_kp],
                               dtype=np.float16)
        # 32768 is enough
        num_person_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, kps):
            person_ind = num_person_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            num_person_frame[frame_ind] += 1

        kpgrp = self.kpsubset
        weight = self.keypoint_weight
        results['num_person'] = num_person

        if num_person > self.max_person:
            for i in range(total_frames):
                np_frame = num_person_frame[i]
                val = new_kpscore[:np_frame, i]

                val = (
                    np.sum(val[:, kpgrp['face']], 1) * weight['face'] +
                    np.sum(val[:, kpgrp['torso']], 1) * weight['torso'] +
                    np.sum(val[:, kpgrp['limb']], 1) * weight['limb'])
                inds = sorted(range(np_frame), key=lambda x: -val[x])
                new_kpscore[:np_frame, i] = new_kpscore[inds, i]
                new_kp[:np_frame, i] = new_kp[inds, i]
            results['num_person'] = self.max_person

        results['keypoint'] = new_kp[:self.max_person]
        results['keypoint_score'] = new_kpscore[:self.max_person]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'squeeze={self.squeeze}, '
                    f'max_person={self.max_person}, '
                    f'keypoint_weight={self.keypoint_weight}, '
                    f'source={self.source}, '
                    f'kwargs={self.kwargs})')
        return repr_str

@PIPELINES.register_module()
class GeneratePoseHistoryMotionTarget:
    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 # skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                 #            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                 #            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 11), (6, 12), (11, 12), (5, 7),
                            (7, 9), (6, 8), (8, 10), (11, 13),
                            (13, 15), (12, 14), (14, 16)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 num_joints=17):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.num_joints =num_joints
    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):

            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:

            num_kp = kps.shape[1] #11
            for i in range(num_kp):
                # sigma =0.6
                # if i in [5,6,7,8]:
                #     sigma =2
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                heatmaps.append(heatmap)
            if self.num_joints==19:
                #add heatmaps for neck and middle hip joints
                neck_joint = (kps[:, 5]+kps[:, 6])/2
                neck_joint_conf = (max_values[:, 5]+max_values[:, 6])/2
                middle_hip_joint = (kps[:, 11]+kps[:, 12])/2
                middle_hip_joint_conf = (max_values[:, 11] + max_values[:, 12]) / 2
                for joint,conf in zip([neck_joint,middle_hip_joint],[neck_joint_conf,middle_hip_joint_conf]):
                    heatmap = self.generate_a_heatmap(img_h, img_w, joint,
                                                      sigma, conf)
                    heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def generate_pmhi_per_joint(self,sklJoints_images):
        """
        Generates a motion history image based on array of binary images.
        Args:
            sklJoints_images (list): List of binary images
        Returns:
            m_t (np array): Motion history image
        """
        m_t = {}
        for i in range(17):
            m_t[i] = sklJoints_images[0][:, :, i]
        for i in range(1, len(sklJoints_images)):
            frame_heatmaps = sklJoints_images[i]
            for counter in range(17):
                m_t[counter] = m_t[counter] * 0.95 + frame_heatmaps[:, :, counter]
        for i in range(17):
            m_t[i] = np.clip(m_t[i], 0, 255)
            #m_t[i] -= m_t[i].min()
            #m_t[i] /= m_t[i].max()
        print("list(m_t.values())",list(m_t.keys()))
        m_t=np.stack(list(m_t.values()), axis=-1)
        return m_t
    def generate_pmhi(self,sklJoints_images):
        """
        Generates a motion history image based on array of binary images.
        Args:
            sklJoints_images (list): List of binary images
        Returns:
            m_t (np array): Motion history image
        """
        m_t = sklJoints_images[0]
        print("len(sklJoints_images)", len(sklJoints_images))
        for x in range(1, len(sklJoints_images)):
            m_t = m_t * 0.95 + sklJoints_images[x]

        print("max and min")
        print(np.max(m_t), np.min(m_t))
        m_t = np.clip(m_t, 0, 255)
        m_t -= m_t.min()
        m_t /= m_t.max()
        print(np.max(m_t), np.min(m_t))
        return m_t
    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        imgs = []
        prev_kps =None
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            #print(kps.shape)
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        #---generate pose history image for each joint across all frames ---#
        segment_len = (num_frame//8)
        pmhImgs =[]
        start_segment,end_segment =0,0
        for i in range(8):
            start_segment = end_segment
            if i >0:
                start_segment -= (segment_len//2)
            end_segment += segment_len
            if i==7:
                end_segment = num_frame
            #print(start_segment,end_segment)
            phMotion=self.generate_pmhi_per_joint(imgs[start_segment:end_segment])
            pmhImgs.append(phMotion)
        return pmhImgs
    def __call__(self, results):
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str
@PIPELINES.register_module()
class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 num_joints=17):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.num_joints =num_joints

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):

            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_a_body_part_limb_heatmap(self,img_h, img_w, starts_part,
                                      ends_part, sigma,
                                      start_values_part,
                                      end_values_part):

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        #print("len(starts_part)",len(starts_part))
        for idx in range(len(starts_part)):
            starts = starts_part[idx]
            ends = ends_part[idx]
            start_values=start_values_part[idx]
            end_values= end_values_part[idx]
            for start, end, start_value, end_value in zip(starts, ends,
                                                          start_values,
                                                          end_values):
                value_coeff = min(start_value, end_value)
                if value_coeff < self.eps:
                    continue

                min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
                min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

                min_x = max(int(min_x - 3 * sigma), 0)
                max_x = min(int(max_x + 3 * sigma) + 1, img_w)
                min_y = max(int(min_y - 3 * sigma), 0)
                max_y = min(int(max_y + 3 * sigma) + 1, img_h)

                x = np.arange(min_x, max_x, 1, np.float32)
                y = np.arange(min_y, max_y, 1, np.float32)

                if not (len(x) and len(y)):
                    continue

                y = y[:, None]
                x_0 = np.zeros_like(x)
                y_0 = np.zeros_like(y)

                # distance to start keypoints
                d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

                # distance to end keypoints
                d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

                # the distance between start and end keypoints.
                d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

                if d2_ab < 1:
                    full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                       sigma, [start_value])
                    heatmap = np.maximum(heatmap, full_map)
                    continue

                coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

                a_dominate = coeff <= 0
                b_dominate = coeff >= 1
                seg_dominate = 1 - a_dominate - b_dominate

                position = np.stack([x + y_0, y + x_0], axis=-1)
                projection = start + np.stack([coeff, coeff], axis=-1) * (
                        end - start)
                d2_line = position - projection
                d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
                d2_seg = (
                        a_dominate * d2_start + b_dominate * d2_end +
                        seg_dominate * d2_line)

                patch = np.exp(-d2_seg / 2. / sigma ** 2)
                patch = patch * value_coeff

                heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                    heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap
    #Ahmed Abdelkawy
    def generate_heatmap_modified(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                # sigma =0.6
                # if i in [5,6,7,8]:
                #     sigma =2
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma[i], max_values[:, i])
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def generate_part_heatmap(self, img_h, img_w, centers_lst, sigma, max_values_lst):
        """Generate pseudo heatmap for one keypoint in one frame.
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        for prt_idx in range(len(centers_lst)):
            centers, max_values = centers_lst[prt_idx], max_values_lst[prt_idx]
            for center, max_value in zip(centers, max_values):
                mu_x, mu_y = center[0], center[1]
                if max_value < self.eps:
                    continue

                st_x = max(int(mu_x - 3 * sigma), 0)
                ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
                st_y = max(int(mu_y - 3 * sigma), 0)
                ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
                x = np.arange(st_x, ed_x, 1, np.float32)
                y = np.arange(st_y, ed_y, 1, np.float32)

                # if the keypoint not in the heatmap coordinate system
                if not (len(x) and len(y)):
                    continue
                y = y[:, None]

                patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
                patch = patch * max_value
                heatmap[st_y:ed_y,
                st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                        patch)

        return heatmap
    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        # =====get shared bbox across frames ===#

        heatmaps = []
        if self.with_kp:
            number_parts = 5
            # parts_kps = [(6, 8, 10), (5, 7, 9), (0, 1, 2, 3, 4), (12, 14, 16), (11, 13, 15)]
            # for i in range(number_parts):
            #     # get kps of specific part
            #     kps_part, max_values_part = [], []
            #     for idx in parts_kps[i]:
            #         kps_part.append(kps[:, idx])
            #         max_values_part.append(max_values[:, idx])
            #
            #     heatmap = self.generate_part_heatmap(img_h, img_w, kps_part,
            #
            #                                       sigma, max_values_part)
            #     heatmaps.append(heatmap)
            num_kp = kps.shape[1] #11
            for i in range(num_kp):
                # sigma =0.6
                # if i in [5,6,7,8]:
                #     sigma =2
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])

                #heatmap =self.resize_image(heatmap[y:y2, x:x2], size=(56, 56))
                heatmaps.append(heatmap)
            if self.num_joints==19:
                #add heatmaps for neck and middle hip joints
                neck_joint = (kps[:, 5]+kps[:, 6])/2
                neck_joint_conf = (max_values[:, 5]+max_values[:, 6])/2
                middle_hip_joint = (kps[:, 11]+kps[:, 12])/2
                middle_hip_joint_conf = (max_values[:, 11] + max_values[:, 12]) / 2
                for joint,conf in zip([neck_joint,middle_hip_joint],[neck_joint_conf,middle_hip_joint_conf]):
                    heatmap = self.generate_a_heatmap(img_h, img_w, joint,
                                                      sigma, conf)
                    heatmaps.append(heatmap)

        if self.with_limb:
            # skelet = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
            #              (7, 9), (0, 6), (6, 8), (8, 10))
            number_parts = 5
            # parts_kps = [[(6, 8), (8, 10)], [(5, 7), (7, 9)],
            #              [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (6, 12), (5, 11), (11, 12)],
            #              [(12, 14), (14, 16)], [(11, 13), (13, 15)]]
            # for i in range(number_parts):
            #     # get kps of specific part
            #     starts_part, ends_part, start_values_part, end_values_part = [], [], [], []
            #     for limb in parts_kps[i]:
            #         start_idx, end_idx = limb
            #         starts_part.append(kps[:, start_idx])
            #         ends_part.append(kps[:, end_idx])
            #
            #         start_values_part.append(max_values[:, start_idx])
            #         end_values_part.append(max_values[:, end_idx])
            #
            #     heatmap = self.generate_a_body_part_limb_heatmap(img_h, img_w, starts_part,
            #                                                      ends_part, sigma,
            #                                                      start_values_part,
            #                                                      end_values_part)
            #     heatmaps.append(heatmap)
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)


        #heatmaps = [self.resize_image(cp.deepcopy(hmap[y:y2, x:x2]), size=(56, 56)) for hmap in heatmaps]

        return np.stack(heatmaps, axis=-1)

    def add_joint_attention_upper(self,kps, prev_kps):
        num_joints =51
        sigma = 0.6 * np.ones((num_joints, 1), dtype=np.float32)
        joints_pair_lst = [(5,7),(7,9),(6,8),(8,10)]
        for i,j in joints_pair_lst:
            #compute attention for left hand
            # calculating Euclidean distance
            # using linalg.norm()
            point1=kps[0, i, :]
            point2 =kps[0,j,:]
            cur_dist = np.linalg.norm(point1 - point2)
            point1 = prev_kps[0, i, :]
            point2 = prev_kps[0, j, :]
            prev_dist = np.linalg.norm(point1 - point2)
            #if abs(prev_dist-cur_dist) > 3:
            sigma[j] =0.6+abs(prev_dist-cur_dist)
        return sigma
    def add_joint_attention(self,num_joints,kps, prev_kps):
        sigma = 0.6 * np.ones((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            #compute attention for left hand
            # calculating Euclidean distance
            # using linalg.norm()
            point1=kps[0, i, :]
            point2 = prev_kps[0, i, :]
            if np.all(point1==0) or np.all(point2==0):
                dist =0
            else:
                dist = np.linalg.norm(point1 - point2)
            #if abs(prev_dist-cur_dist) > 3:

            sigma[i] = 0.6
            if dist >0:
                sigma[i] =0.6+np.log(1+dist)
            #print(dist)
            # if sigma[i] > 6:
            #     sigma[i]=2
        return sigma
    #Ahmed Abdelkawy
    def gen_an_aug_modified(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        imgs = []
        prev_kps =None
        num_joints =kp_shape[2]
        # print("all_kps.shape",all_kps.shape,all_kps[:, 0].shape)
        # exit(-1)
        for i in range(num_frame):
            sigma = 0.6 * np.ones((num_joints, 1), dtype=np.float32)
            #sigma = self.sigma
            kps = all_kps[:, i]

            #print(kps.shape,kps[0,1,:].tolist())

            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            if i >=1: #add hard attention here
                sigma = self.add_joint_attention(num_joints,kps,prev_kps)
            prev_kps =kps
            hmap = self.generate_heatmap_modified(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        return imgs



    def resize_image(self,img, size=(28, 28)):
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        if h == w:
            return cv2.resize(img, size, cv2.INTER_AREA)

        dif = h if h > w else w
        interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2
        heatmaps =[]
        for i in range(c):
            mask = np.zeros((dif, dif), dtype=np.float32)
            mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w,i]
            heatmaps.append(cv2.resize(mask, size, interpolation))
        # else:
        #     mask = np.zeros((dif, dif, c), dtype=img.dtype)#*np.array([85,12,72])
        #     #mask = np.full((dif, dif, c), [0, 0, 0], dtype=img.dtype)  # [82,0,66]
        #     mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

        return np.stack(heatmaps, axis=-1)
    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape
        # Make NaN zero
        all_kps[np.isnan(all_kps)] = 0.

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        # # =====get shared bbox across frames ===#
        # x, y, x2, y2 = results['min_bbox']
        # denom = 4
        # x, y, x2, y2 = int(x / denom), int(y / denom), int(x2 / denom), int(y2 / denom)

        imgs = []
        prev_kps =None
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            #print(kps.shape)
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)
            #imgs.append(cv2.resize(hmap[y:y2, x:x2], (56, 56)))
            #imgs.append(self.resize_image(hmap[y:y2, x:x2], size=(56, 56)))

        return imgs
    def __call__(self, results):
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'
        if not self.double:
            results[key] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results[key] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


@PIPELINES.register_module()
class PaddingWithLoop:
    """Sample frames from the video.

    To sample an n-frame clip from the video, PaddingWithLoop samples
    the frames from zero index, and loop the frames if the length of
    video frames is less than te value of 'clip_len'.

    Required keys are "total_frames", added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
    """

    def __init__(self, clip_len, num_clips=1):

        self.clip_len = clip_len
        self.num_clips = num_clips

    def __call__(self, results):
        num_frames = results['total_frames']

        start = 0
        inds = np.arange(start, start + self.clip_len)
        inds = np.mod(inds, num_frames)

        results['frame_inds'] = inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class PoseNormalize:
    """Normalize the range of keypoint values to [-1,1].

    Args:
        mean (list | tuple): The mean value of the keypoint values.
        min_value (list | tuple): The minimum value of the keypoint values.
        max_value (list | tuple): The maximum value of the keypoint values.
    """

    def __init__(self,
                 mean=(960., 540., 0.5),
                 min_value=(0., 0., 0.),
                 max_value=(1920, 1080, 1.)):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.min_value = np.array(
            min_value, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.max_value = np.array(
            max_value, dtype=np.float32).reshape(-1, 1, 1, 1)

    def __call__(self, results):
        keypoint = results['keypoint']
        keypoint = (keypoint - self.mean) / (self.max_value - self.min_value)
        results['keypoint'] = keypoint
        results['keypoint_norm_cfg'] = dict(
            mean=self.mean, min_value=self.min_value, max_value=self.max_value)
        return results
