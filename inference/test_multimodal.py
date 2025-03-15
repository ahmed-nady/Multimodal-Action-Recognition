import random
import torch
from actionModels.rgb_pose_action_recognition_model import RGBActionRecognition,PoseActionRecognition,\
    RGBPoseAttentionActionRecognizer
import pickle
import core.utils as utils
from collections import OrderedDict
import re
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler

import numpy as np
from mmcv import load
import torch
import mmcv
import dataPrep.dataPrepRGB as dataPrepRGB
import dataPrep.dataPrep as dataPrep
from torch.utils.data import DataLoader
from dataPrep.dataPrepMultiModality import CustomPoseDataset
from core.evalMetrics import EvalMetrics
from core.log_buffer import LogBuffer
from functools import partial

dataset= 'toyota'
evaluation_protocol ='xsub'
def parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')


        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_value = loss_value.data.clone()
                # print("torch.distributed.get_world_size()",torch.distributed.get_world_size())
                torch.distributed.all_reduce(loss_value.div_(torch.distributed.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return log_vars

if dataset=='ntu60':

    if evaluation_protocol=='xsub':
        ann_file_train = 'ntu60_xsub_train.pkl' 
        ann_file_val = 'ntu60_xsub_val.pkl'
    else:
        ann_file_train = 'ntu60_xview_train.pkl'  
        ann_file_val = 'ntu60_xview_val.pkl'
elif dataset=='ntu120':
        if evaluation_protocol == 'xsub':
            ann_file_train = ntu120_xsub_train.pkl' 
            ann_file_val = 'ntu120_xsub_val.pkl'
        else:
            ann_file_train = '/ntu120_xset_train.pkl'
            ann_file_val = 'ntu120_xset_val.pkl'
elif dataset == 'toyota':
    if evaluation_protocol == 'xsub':
        ann_file_train = 'train_CS.pkl'  # posec3d_witout_fingers
        ann_file_val = 'test_CS.pkl'
    else:
        ann_file_train = 'train_CV2.pkl'  # posec3d_witout_fingers
        ann_file_val = 'test_CV2.pkl'

print("evaluation_protocol",evaluation_protocol)

validation_annos = load(ann_file_val)

test_data = CustomPoseDataset(dataset,validation_annos, mode='test',pose_input='joint')
# test_dataLoader = DataLoader(test_data, batch_size, shuffle=False, num_workers=1,
#     #                              sampler=DistributedSampler(test_data))


def evaluate(model, test_dataLoader,batch_size,gpu_id=0):
    model.eval()
    stepsNo = len(test_dataLoader)
    results =[]
    prog_bar = mmcv.ProgressBar(stepsNo*batch_size)
    test_log_buffer = LogBuffer()
    evalMetric = EvalMetrics()
    with torch.no_grad():
        losses = dict()
        model_preds = []
        gt_labels = []
        for iterationNo, (imgs, headmap_vols, targets) in enumerate(test_dataLoader):
            imgs = imgs.to(gpu_id)
            headmap_vols = headmap_vols.to(gpu_id)
            targets = targets.to(gpu_id)
            #rgb_preds,pose_preds  = model(imgs.reshape((-1,)+imgs.shape[2:]),headmap_vols.reshape((-1,)+headmap_vols.shape[2:]))
            rgb_preds,pose_preds  = model(imgs,headmap_vols)

            rgb_pose_preds = average_clip(rgb_preds+pose_preds,num_segs=num_clips,average_clips=average_clips)

            rgb_scores = rgb_preds.detach().cpu().numpy()
            pose_scores = pose_preds.detach().cpu().numpy()
            preds_np = rgb_scores+pose_scores
            labels_np =targets.cpu().numpy()

            model_preds.extend(preds_np)
            gt_labels.extend(labels_np)

            results.extend(preds_np)
            batch_size = len(preds_np)
            for _ in range(batch_size):
                prog_bar.update()
            # preds_np, labels_np = preds.detach().cpu().numpy(), targets.cpu().numpy()
            top_k_acc = evalMetric.top_k_accuracy(preds_np, labels_np, topk=(1, 5))
            if top_k_acc[0]<1.0:
                print(top_k_acc)
            mean_class_accuracy = evalMetric.mean_class_accuracy(preds_np, labels_np)
            # top_k_acc = self.evalMetric.top_k_accuracy(preds.detach().cpu().numpy(), targets.cpu().numpy(), topk=(1, 5))

            for k, a in zip((1, 5), top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=imgs.device)
            losses['mean_class_acc'] = torch.tensor(mean_class_accuracy, device=imgs.device)
            log_vars = parse_losses(losses)
            test_log_buffer.update(log_vars, imgs.shape[0])


        test_log_buffer.average(stepsNo)
        top1_acc, top5_acc, mean_acc = test_log_buffer.output['top1_acc'], test_log_buffer.output[
            'top5_acc'], \
            test_log_buffer.output['mean_class_acc']
        test_log_buffer.clear()

        #==========================================#
        model_preds = np.asarray(model_preds)
        gt_labels = np.asarray(gt_labels)
        top_k_acc = evalMetric.top_k_accuracy(model_preds, gt_labels, topk=(1, 5))
        mean_class_accuracy = evalMetric.mean_class_accuracy(model_preds, gt_labels)
        print(f"\ntop_k_acc: {top_k_acc} == mean_acc: {mean_class_accuracy}")

        #=============================================#
        msg = f"Evaluating top_k_accuracy ...\n top1_acc: {top1_acc:.5f}\n top5_acc: {top5_acc:.5f}\n Evaluating mean_class_accuracy ...\n mean_acc :{mean_acc:.5f}"
        print(msg)

    return results
def load_checkpoint(model,model_checkpoint,gpu_id):
    loc = f"cuda:{gpu_id}"
    checkpoint = torch.load(model_checkpoint,map_location=loc)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    import re
    revise_keys = [(r'^module\.', '')]
    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    model.load_state_dict(state_dict, strict=True)
    print('Loaded Successfully...!')

def average_clip(cls_score, num_segs=1,average_clips=None):
    """Averaging class score over multiple clips.

    Using different averaging types ('score' or 'prob' or None,
    which defined in test_cfg) to computed the final averaged
    class score. Only called in test mode.

    Args:
        cls_score (torch.Tensor): Class score to be averaged.
        num_segs (int): Number of clips for each input sample.

    Returns:
        torch.Tensor: Averaged class score.
    """

    #average_clips = 'prob'
    if average_clips not in ['score', 'prob', None]:
        raise ValueError(f'{average_clips} is not supported. '
                         f'Currently supported ones are '
                         f'["score", "prob", None]')

    if average_clips is None:
        return cls_score

    batch_size = cls_score.shape[0]
    cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

    if average_clips == 'prob':
        cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
    elif average_clips == 'score':
        cls_score = cls_score.mean(dim=1)

    return cls_score
def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

if __name__=='__main__':
    utils.set_random_seed(seed=100, deterministic=True)
    batch_size = 160
    gpu_id =1
    num_clips=1
    average_clips =None#'prob'

    out_pkl_file = 'results/multimodal_x3dTShiftRGB_X3dTShiftPose_' + dataset + '_' + evaluation_protocol + '.pkl'

    test_dataLoader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)
    setting = dataset + '_' + evaluation_protocol
    number_classes =None
    if setting == 'ntu60_xsub':
        multimodal_checkpoint = '/pretrained-EPAM_models/X3dRGBTShift_X3dPoseTShift_double_shifted_chs_CBAM_spatial_efficient_temporal_NTU60_XSub_best_top1_acc_epoch_4.pth'
        number_classes=60
    elif setting == 'ntu60_xview':
        multimodal_checkpoint = '/pretrained-EPAM_models/X3dRGBTShift_X3dPoseTShift_double_shifted_chs_CBAM_spatial_efficient_temporal_NTU60_XView_best_top1_acc_8.pth'
        number_classes = 60
    elif setting == 'ntu120_xsub':
        multimodal_checkpoint ='/pretrained-EPAM_models/X3dRGBTShift_X3dPoseTShift_double_shifted_chs_CBAM_spatial_efficient_temporal_NTU120_XSub_best_top1_acc_8.pth'
        number_classes = 120
    elif setting == 'ntu120_xset':
        multimodal_checkpoint = '/pretrained-EPAM_models/X3dRGBTShift_X3dPoseTShift_double_shifted_chs_CBAM_spatial_efficient_temporal_NTU120_XSet_best_top1_acc_10.pth'
        number_classes = 120
    elif setting =='toyota_xsub':
        multimodal_checkpoint ='/pretrained-EPAM_models/X3dRGBTShift_X3dPoseTShift_double_shifted_chs_CBAM_spatial_efficient_temporal_ToyotaSH_XSub_best_top1_acc_epoch_11.pth'
        number_classes = 31
    # Constructing the DDP model
    model = RGBPoseAttentionActionRecognizer(attention ='CBAM_spatial_efficient_temporal',backbone_type='poseX3dTShiftSE',num_classes=number_classes)
    load_checkpoint(model, multimodal_checkpoint,gpu_id)

    model = model.to(gpu_id)
    # model.init_weights()
    results = evaluate(model, test_dataLoader, batch_size, gpu_id)
    # dump_results

    with open(out_pkl_file, 'wb') as f:
        pickle.dump(results, f)
