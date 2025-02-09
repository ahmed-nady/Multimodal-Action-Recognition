import random
import torch
from actionModels.rgb_pose_action_recognition_model import RGBActionRecognition,PoseActionRecognition
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
from core.evalMetrics import EvalMetrics
from core.log_buffer import LogBuffer

dataset='ntu60'
rgb_or_pose ='pose'
evaluation_protocol ='xview'
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
    num_classes =60
    if evaluation_protocol=='xsub':
        if rgb_or_pose == 'rgb':
            ann_file_train = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xsub_train.txt'
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xsub_val.txt'
            ann_file_test = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xsub_val.txt'
        else:
            ann_file_train = '/home/a0nady01/ActionRecognition/mmaction2/Action Recognition/ICIP_paper/ntu60_xview_pose_sim_train.pkl'
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/Action Recognition/ICIP_paper/ntu60_xsub_pose_sim_val.pkl'
            ann_file_test = '/home/a0nady01/ActionRecognition/mmaction2/Action Recognition/ICIP_paper/ntu60_xsub_pose_sim_val.pkl'

    else:
        if rgb_or_pose == 'rgb':
            ann_file_train = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xview_train.txt'
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xview_val.txt'
            ann_file_test = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu60_xview_val.txt'
        else:
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/ntu60_xview_val.pkl'
elif dataset=='ntu120':
    num_classes = 120
    if evaluation_protocol == 'xsub':
        if rgb_or_pose == 'rgb':
            ann_file_train = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xsub_train.txt'
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xsub_val.txt'
            ann_file_test = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xsub_val.txt'
        else:
            pass
    else:
        if rgb_or_pose == 'rgb':
            ann_file_train = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xset_train.txt'
            ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xset_val.txt'
            ann_file_test = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/HumanParsing/ntu120_xset_val.txt'

elif dataset=='pku':
    num_classes = 51
    if evaluation_protocol=='xsub':
        ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/PKU-MMD/pku_xsub_test.pkl'
    else:
        ann_file_val = '/home/a0nady01/ActionRecognition/mmaction2/CustomLearning/PKU-MMD/pku_xview_test.pkl'

print("evaluation_protocol",evaluation_protocol)

if rgb_or_pose == 'rgb':
    test_data = dataPrepRGB.CustomPoseDataset(ann_file_val, mode='test')
else:
    validation_ntu_annos = load(ann_file_val)
    test_data = dataPrep.CustomPoseDataset(validation_ntu_annos, mode='test')


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
        for iterationNo, (inputs, targets) in enumerate(test_dataLoader):
            inputs = inputs.to(gpu_id)
            targets = targets.to(gpu_id)
            preds = model(inputs)
            preds_np, labels_np = preds.detach().cpu().numpy(), targets.cpu().numpy()

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
                    a, device=inputs.device)
            losses['mean_class_acc'] = torch.tensor(mean_class_accuracy, device=inputs.device)
            log_vars = parse_losses(losses)
            test_log_buffer.update(log_vars, inputs.shape[0])


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
def load_checkpoint(model,model_checkpoint,rgb=True):
    print(f"Looad checkpoint from {model_checkpoint}")
    checkpt = torch.load(model_checkpoint, map_location='cpu')
    # get state_dict from checkpoint
    if 'state_dict' in checkpt:
        state_dict = checkpt['state_dict']
    elif 'model_state_dict' in checkpt:
        state_dict = checkpt['model_state_dict']
    else:
        state_dict = checkpt

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict_mod = OrderedDict()
    if not rgb:
        for k, v in state_dict.items():
            if 'module.pose_backbone' in k:
                state_dict_mod[re.sub('module.pose_backbone.', 'pose_backbone.', k)] = v
            elif 'pose_backbone' in k:
                state_dict_mod[re.sub('pose_backbone.', '', k)] = v
            if 'backbone' in k:
                state_dict_mod[re.sub('backbone.', 'backbone.', k)] = v
            elif 'module.pose_cls_head' in k:
                state_dict_mod[re.sub('module.pose_cls_head.', '', k)] = v
            elif 'pose_cls_head' in k:
                state_dict_mod[re.sub('pose_cls_head.', '', k)] = v
            elif 'cls_head' in k:
                state_dict_mod[re.sub('cls_head.', 'cls_head.', k)] = v
    else:
        for k, v in state_dict.items():
            if 'module.rgb_backbone' in k:
                state_dict_mod[re.sub('module.rgb_backbone.', '', k)] = v
            elif 'rgb_backbone' in k:
                state_dict_mod[re.sub('rgb_backbone.', '', k)] = v
            elif 'backbone' in k:
                state_dict_mod[re.sub('backbone.', 'rgb_backbone.', k)] = v
            elif 'module.rgb_cls_head' in k:
                state_dict_mod[re.sub('module.rgb_cls_head.', '', k)] = v
            elif 'rgb_cls_head' in k:
                state_dict_mod[re.sub('rgb_cls_head.', '', k)] = v
            elif 'cls_head' in k:
                state_dict_mod[re.sub('cls_head.', 'rgb_cls_head.', k)] = v

    # # Keep metadata in state_dict
    state_dict_mod._metadata = metadata
    model.load_state_dict(state_dict_mod, strict=True)
    print('Loaded Successfully...!')
    


if __name__=='__main__':
    batch_size = 128
    gpu_id =0

    if rgb_or_pose =='rgb':
        out_pkl_file = 'results/x3dRGB_'+dataset+'_'+evaluation_protocol+'_epoch_200.pkl'
        test_dataLoader = DataLoader(test_data, batch_size, shuffle=False, num_workers=1)
    else:
        #out_pkl_file = 'out_c3dLaterality_ntu60_xsub.pkl'
        #out_pkl_file = 'analysis/out_poseX3d_SE_ntu60_xsub.pkl'
        out_pkl_file = 'results/x3dTShiftPose_d_chs_'+dataset+'_'+evaluation_protocol+'_best_preds_56x56.pkl'
        test_dataLoader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)

    utils.set_random_seed(seed=100, deterministic=True)
    # Constructing the DDP model
    if rgb_or_pose == 'rgb':
        setting = dataset+'_'+evaluation_protocol
        if setting=='ntu60_xsub':
            rgb_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/RGB/ntu60_XSub_X3dRGB_ME_v2_epoch_180.pth'
            rgb_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/RGB/ntu60_XSub_X3dRGB_ME_v2_epoch_195.pth'
            rgb_checkpoint ='/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/RGB/epoch_200.pth'
        elif setting=='ntu60_xview':
            rgb_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XView/RGB/ntu60_xview_X3dRGB_ME_epoch_195.pth'
            rgb_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XView/RGB/epoch_190.pth'
        elif setting =='ntu120_xsub':
            rgb_checkpoint ='/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSub/RGB/ntu120_xsub_x3dRGB_ME_epoch_205.pth'
            rgb_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSub/RGB/epoch_205.pth'
        model = RGBActionRecognition(num_classes=120)
        load_checkpoint(model, rgb_checkpoint)
    else:
       
        setting = dataset + '_' + evaluation_protocol
        if setting == 'pku_xsub':
            pose_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/PKU/XSub/Pose/pku_xsub_XTShiftPose_SE_double_chs_best_top1_acc_epoch_240.pth'
        elif setting == 'pku_xview':
            pose_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/PKU/XView/Pose/pku_xview_XTShiftPose_SE_double_chs_best_top1_acc_epoch_240.pth'
        elif setting=='ntu60_xsub':
            pose_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/Pose/ntu60_xsub_x3dTShiftPose_double_shifted_chs_best_top1_acc_epoch_235.pth'
        elif setting == 'ntu60_xview':
            pose_checkpoint = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/spatialTemporalAlignment/NTU60/XView/Pose/ntu60_xview_x3dTShiftPose_double_shifted_chs_best_top1_acc_epoch_230.pth'
        model = PoseActionRecognition(backbone_type='poseX3dTShiftSE',num_classes=num_classes,conv1_stride=1,num_stages=3)#backbone_type = C3DLateralityPartSubnetFusion or c3d
        load_checkpoint(model, pose_checkpoint,rgb=False)
    model = model.to(gpu_id)
    #model.init_weights()
    results = evaluate(model,test_dataLoader,batch_size,gpu_id)
    #dump_results

    with open(out_pkl_file, 'wb') as file:
        pickle.dump(results, file)
