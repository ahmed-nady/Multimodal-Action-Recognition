import os
from collections import OrderedDict
import numpy as np
import re

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.cnn import kaiming_init,constant_init
from mmcv.utils import _BatchNorm
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from core.losses import LabelSmoothingLoss

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataLoader: DataLoader,
                 test_dataLoader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 loss_fn,
                 evalMetric,
                 gpu_id: int,
                 sWriter,
                 log_buffer,
                 test_log_buffer,
                 save_every: int,
                 evaluate_every: int,
                 logger,
                 work_dir,
                 mode='multi-gpus',
                 pretrained=False
                 ) -> None:
        self.mode= mode
        if mode == 'multi-gpus':
            self.gpu_id = gpu_id
        elif mode=='multi-nodes':
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
        else:
            self.gpu_id =0
        self.model = model.to(self.gpu_id)
        self.train_dataLoader = train_dataLoader
        self.test_dataLoader = test_dataLoader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.evalMetric = evalMetric
        self.save_every = save_every
        self.evaluate_every = evaluate_every
        self.sWriter = sWriter
        self.log_interval = 20
        self.best_ck_pt_path = None
        self.best_top_1_acc = 0
        self.best_top_1_epoch =0

        self.work_dir = work_dir
        self.logger = logger
        self.log_buffer = log_buffer
        self.test_log_buffer = test_log_buffer
        # self.PosePretrained = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/pose_best_top1_acc_epoch_330.pth'
        # self.RGBPretrained = '/home/a0nady01/ActionRecognition/mmaction2/work_dirs/RGBPosePretrained/rgb_best_top1_acc_epoch_235.pth'

        if pretrained:
            self.init_weights()
        # initialize model
        if mode !='single-gpu':
            self.model = DDP(self.model, device_ids=[self.gpu_id])#,find_unused_parameters=True

    def init_weights(self):
        print('init super class')
    
    def _run_batch(self, source, targets, epochIterationNo):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        self.sWriter.add_scalar('Loss/Train', loss, epochIterationNo)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    # ===borrowed from mmaction2=========
    @staticmethod
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

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_value = loss_value.data.clone()
                # print("torch.distributed.get_world_size()",torch.distributed.get_world_size())
                torch.distributed.all_reduce(loss_value.div_(torch.distributed.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return log_vars

    def _run_epoch(self, epoch):

        b_sz = len(next(iter(self.train_dataLoader))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dataLoader)}")
        if self.mode !='single-gpu':
            self.train_dataLoader.sampler.set_epoch(epoch)

        losses = dict()
        num_iterations = len(self.train_dataLoader)
        for iterationNo, (inputs, targets) in enumerate(self.train_dataLoader):
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            # self._run_batch(source, targets,epoch+self.iterationNo)
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            loss_value = F.cross_entropy(preds, targets)
            loss_value.backward()
            # ====gradient clipping====
            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=40, norm_type=2.)
            # ==============================#

            self.optimizer.step()
            self.scheduler.step()
            # ===calculate top-1, top-5 acc
            output_np, targets_np = preds.detach().cpu().numpy(), targets.cpu().numpy()
            top_k_acc = self.evalMetric.top_k_accuracy(output_np, targets_np, topk=(1, 5))
            mean_class_accuracy = self.evalMetric.mean_class_accuracy(output_np, targets_np)
            for k, a in zip((1, 5), top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=loss_value.device)
            losses['loss'] = loss_value
            losses['mean_class_acc'] = torch.tensor(mean_class_accuracy, device=loss_value.device)
            log_vars = self.parse_losses(losses)
            self.log_buffer.update(log_vars, inputs.shape[0])

            if iterationNo % self.log_interval == 0: #and self.gpu_id == 0:
                self.log_buffer.average(self.log_interval)
                # print(log_buffer.output)
                top1_acc, top5_acc, mean_acc, loss = self.log_buffer.output['top1_acc'], self.log_buffer.output[
                    'top5_acc'], \
                    self.log_buffer.output['mean_class_acc'], self.log_buffer.output['loss']
                self.log_buffer.clear()
                # topK_acc=self.evalMetric.top_k_accuracy(preds ,np.array(labels),topk=(1,5))
                # ====log loss and acc====
                self.sWriter.add_scalar('Train/loss', loss, epoch * num_iterations + iterationNo)
                self.sWriter.add_scalar('Train/top1_acc', top1_acc, epoch * num_iterations + iterationNo)
                self.sWriter.add_scalar('Train/top5_acc', top5_acc, epoch * num_iterations + iterationNo)
                self.sWriter.add_scalar('Train/mean_class_acc', mean_acc, epoch * num_iterations + iterationNo)

                msg = f"Epoch: [{epoch}][{iterationNo}|{len(self.train_dataLoader)}] loss: {loss:.3f} top1_acc: {top1_acc:.3f} top5_acc: {top5_acc:.3f} mean_acc: {mean_acc:.3f}"
                print(msg)
                self.logger.info(msg)

    def evaluate(self, epoch):
        self.model.eval()
        stepsNo = len(self.test_dataLoader)
        with torch.no_grad():
            num_iterations = len(self.test_dataLoader)
            losses = dict()
            for iterationNo, (inputs, targets) in enumerate(self.test_dataLoader):
                inputs = inputs.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                preds = self.model(inputs)
                test_loss = F.cross_entropy(preds, targets)
                preds_np, labels_np = preds.detach().cpu().numpy(), targets.cpu().numpy()
                top_k_acc = self.evalMetric.top_k_accuracy(preds_np, labels_np, topk=(1, 5))
                mean_class_accuracy = self.evalMetric.mean_class_accuracy(preds_np, labels_np)
                # top_k_acc = self.evalMetric.top_k_accuracy(preds.detach().cpu().numpy(), targets.cpu().numpy(), topk=(1, 5))

                for k, a in zip((1, 5), top_k_acc):
                    losses[f'top{k}_acc'] = torch.tensor(
                        a, device=test_loss.device)
                losses['loss'] = test_loss
                losses['mean_class_acc'] = torch.tensor(mean_class_accuracy, device=test_loss.device)
                log_vars = self.parse_losses(losses)
                self.test_log_buffer.update(log_vars, inputs.shape[0])

        #if self.gpu_id == 0:
        self.test_log_buffer.average(num_iterations)
        top1_acc, top5_acc, mean_acc, loss = self.test_log_buffer.output['top1_acc'], self.test_log_buffer.output[
            'top5_acc'], \
            self.test_log_buffer.output['mean_class_acc'], self.test_log_buffer.output['loss']
        self.test_log_buffer.clear()
        msg = f"Metrics Sync top1_acc: {top1_acc:.3f}, top5_acc: {top5_acc:.3f}, mean_acc :{mean_acc:.3f} loss: {loss:.3f}"
        print(msg)
        self.logger.info(msg)
        self.sWriter.add_scalar('Test/loss', loss, epoch)
        self.sWriter.add_scalar('Test/top1_acc', top1_acc, epoch)
        self.sWriter.add_scalar('Test/top5_acc', top5_acc, epoch)
        self.sWriter.add_scalar('Test/mean_class_acc', mean_acc, epoch)

        if top1_acc > self.best_top_1_acc:
            # ===remove the previous best_ck_pt and save the current one====#            msg = f"Metrics Sync top1_acc, top5_acc, loss: {top1_acc:.3f}, {top5_acc:.3f}, {loss:.3f}"
            self.best_top_1_acc = top1_acc
            self.best_top_1_epoch = epoch
            self._save_checkpoint(epoch, save_best=True)
            # if os.path.exists(self.best_ck_pt_path):
            #     os.remove(self.best_ck_pt_path)

    def _save_checkpoint(self, epoch, save_best=False):
        ckp = self.model.state_dict()
        # base_dir = 'work_dir/test/'
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        ck_pt_name = "checkpoint_epoch_" if save_best else 'best_top1_acc_epoch_'
        PATH = os.path.join(self.work_dir, ck_pt_name + str(epoch) + ".pth")
        if save_best:
            self.best_ck_pt_path = PATH
        saved_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(saved_dict, PATH)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0 and epoch != 0:
                self._save_checkpoint(epoch)
            if epoch % self.evaluate_every == 0:
                self.logger.info(f"evaluation at  epoch:{epoch}")
                self.evaluate(epoch)


class RGBPoseTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, train_dataLoader: DataLoader, test_dataLoader: DataLoader,
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, loss_fn, evalMetric,
                 gpu_id: int, log_buffer, test_log_buffer, save_every: int, evaluate_every: int, logger,
                 work_dir,dataset='ntu60',eval_protocol='xsub', mode='multi-gpus',pretrained=False) -> None:

        self.dataset= dataset
        self.eval_protocol=eval_protocol
        self.setPretrained(self.dataset,self.eval_protocol)
        print("self.PosePretrained",self.PosePretrained)
        print("self.RGBPretrained", self.RGBPretrained)

        super().__init__(model, train_dataLoader, test_dataLoader, optimizer, scheduler, loss_fn, evalMetric, gpu_id,
                         None, log_buffer, test_log_buffer, save_every, evaluate_every, logger, work_dir, mode,pretrained)
        self.bce_loss = torch.nn.BCELoss()
        self.debug = False
        self.adaptive_score=False

       
        if self.gpu_id == 0:
            self.sWriter = SummaryWriter(log_dir=work_dir, flush_secs=60)
            msg = f"PosePretrained:{self.PosePretrained}\n RGBPretrained:{self.RGBPretrained}"
            self.logger.info(msg)
            #self.logger.info(print(model))

    def setPretrained(self,dataset,eval_setting):
        if dataset=='ntu60':
            if eval_setting=='xsub':
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/Pose/ntu60_xsub_x3dTShiftPose_double_shifted_chs_best_top1_acc_epoch_235.pth'
                self.RGBPretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU60/XSub/RGB/ntu60_xsub_x3dTShiftD_I3dHead_RGB_epoch_170.pth'
                
            else:
                #====ntu60 xview=====#

                self.RGBPretrained = '/RGBPosePretrained/spatialTemporalAlignment/NTU60/XView/RGB/epoch_165.pth'
               
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU60/XView/Pose/ntu60_xview_x3dTShiftPose_double_shifted_chs_best_top1_acc_epoch_230.pth'
                
        elif dataset=='ntu120':
            if eval_setting=='xsub':
                
                #==========ntu120 xsub===#
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSub/Pose/ntu120_xsub_x3dTShiftPose_SE_best_top1_acc_epoch_235.pth'
                self.RGBPretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSub/RGB/ntu120_xsub_x3dTShiftD_I3Dhead_RGB_epoch_205.pth'
            else:
                # ==========ntu120 xset===#
                
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSet/Pose/ntu120_xset_x3dTShiftPose_SE_best_top1_acc_epoch_290.pth'
                self.RGBPretrained ='/RGBPosePretrained/spatialTemporalAlignment/NTU120/XSet/RGB/ntu120_xset_x3dTShiftD_RGB_epoch_195.pth'


        elif dataset=='toyota':
            if eval_setting=='xsub':
                #=============***************************toyota datset********
               
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/Toyota/XSub/Pose/toyota_xsub_x3dTShift_doubel_chs_SE_ntu120_best_top1_acc_epoch_95.pth'
                self.RGBPretrained ='/RGBPosePretrained/spatialTemporalAlignment/Toyota/XSub/RGB/toyota_xsub_x3dTShift_RGB_ntu120_epoch_50.pth'

            else:
                self.PosePretrained ='/RGBPosePretrained/spatialTemporalAlignment/Toyota/XView2/Pose/toyota_xview2_x3dTShift_doubel_chs_SE_ntu120_best_top1_acc_epoch_80.pth'
                self.RGBPretrained = '/RGBPosePretrained/spatialTemporalAlignment/Toyota/XView2/RGB/toyota_xview2_x3dTShift_RGB_ntu120_best_top1_acc_epoch_60.pth'

        elif dataset=='pku':
            if eval_setting=='xsub':
                self.PosePretrained = '/RGBPosePretrained/spatialTemporalAlignment/PKU/XSub/Pose/pku_xsub_XTShiftPose_SE_double_chs_best_top1_acc_epoch_240.pth'
                self.RGBPretrained = '/RGBPosePretrained/spatialTemporalAlignment/PKU/XSub/RGB/pku_xsub_X3dTShift_RGB_best_top1_acc_epoch_205.pth'
            else:
                self.PosePretrained = '/RGBPosePretrained/spatialTemporalAlignment/PKU/XView/Pose/pku_xview_XTShiftPose_SE_double_chs_best_top1_acc_epoch_240.pth'
                self.RGBPretrained = '/RGBPosePretrained/spatialTemporalAlignment/PKU/XView/RGB/pku_xview_X3dTShift_RGB_epoch_155.pth'
         
    def prepare_state_dict(self,checkpoint,rgb_flag=True):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        state_dict_mod = OrderedDict()
        prefix_lst =['module.pose_backbone.','pose_backbone.','backbone.','module.pose_cls_head.']
        if not rgb_flag:
            for k, v in state_dict.items():
                if 'module.pose_backbone' in k:
                    state_dict_mod[re.sub('module.pose_backbone.', '', k)] = v
                elif 'pose_backbone' in k:
                    state_dict_mod[re.sub('pose_backbone.', '', k)] = v
                if 'backbone' in k:
                    state_dict_mod[re.sub('backbone.', '', k)] = v
                elif 'module.pose_cls_head' in k:
                    state_dict_mod[re.sub('module.pose_cls_head.', '', k)] = v
                elif 'pose_cls_head' in k:
                    state_dict_mod[re.sub('pose_cls_head.', '', k)] = v
                elif 'cls_head' in k:
                    state_dict_mod[re.sub('cls_head.', '', k)] = v
        else:
            for k, v in state_dict.items():
                if 'module.rgb_backbone' in k:
                    state_dict_mod[re.sub('module.rgb_backbone.', '', k)] = v
                elif 'rgb_backbone' in k:
                    state_dict_mod[re.sub('rgb_backbone.', '', k)] = v
                elif 'backbone' in k:
                    state_dict_mod[re.sub('backbone.', '', k)] = v
                elif 'module.rgb_cls_head' in k:
                    state_dict_mod[re.sub('module.rgb_cls_head.', '', k)] = v
                elif 'rgb_cls_head' in k:
                    state_dict_mod[re.sub('rgb_cls_head.', '', k)] = v
                elif 'cls_head' in k:
                    state_dict_mod[re.sub('cls_head.', '', k)] = v
        # # Keep metadata in state_dict
        state_dict_mod._metadata = metadata

        return state_dict_mod

    def init_weights(self):

        # initialize the remaining blocks
        for m in self.model.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Conv1d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

            elif isinstance(m, nn.Linear):
                """Initiate the parameters from scratch."""
                normal_init(m, std=0.01)
        #====load pretrained model weights======#
        loc = f"cuda:{self.gpu_id}"
        training_framework_dict = {'pose':'AA','rgb':'AA'} #th values can be AA for our custom Action Recognition Framework and mmaction for mmaction framework
        if isinstance(self.PosePretrained,str) and isinstance(self.RGBPretrained,str):
            print(f"Looad checkpoint from {self.PosePretrained} at {self.gpu_id}")
            checkpoint = torch.load(self.PosePretrained,map_location=loc)
            
            state_dict_mod =self.prepare_state_dict(checkpoint,rgb_flag=False)
            self.model.pose_backbone.load_state_dict(state_dict_mod,strict=False)
            self.model.pose_cls_head.load_state_dict(state_dict_mod,strict=False)

            print(f"Looad checkpoint from {self.RGBPretrained} at {self.gpu_id}")
            checkpoint = torch.load(self.RGBPretrained,map_location=loc)
            
            state_dict_mod = self.prepare_state_dict(checkpoint, rgb_flag=True)
            self.model.rgb_backbone.load_state_dict(state_dict_mod, strict=False)
            self.model.rgb_cls_head.load_state_dict(state_dict_mod, strict=False)

            print('Loaded Successfully...!')
            

    def load_checkpoint(self):
        loc = f"cuda:{self.gpu_id}"
         checkpoint = torch.load('/proposedFramework/X3d_m_RGB_epoch_60_X3d_s_Pose_spatial_temporal_alignment_toyota_XSub/best_top1_acc_epoch_10.pth',map_location=loc)
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

        self.model.load_state_dict(state_dict, strict=True)
        print('Loaded Successfully...!')
    def _run_epoch(self, epoch):

        b_sz = len(next(iter(self.train_dataLoader))[0])
        if self.debug:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dataLoader)}")
        self.train_dataLoader.sampler.set_epoch(epoch)

        losses = dict()
        num_iterations = len(self.train_dataLoader)
        #total_loss, num_samples = 0, 0
        for iterationNo, (imgs, headmap_vols, targets) in enumerate(self.train_dataLoader):
            if self.debug:
                print('imgs', imgs.shape)
            imgs = imgs.to(self.gpu_id)
            headmap_vols = headmap_vols.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            # self._run_batch(source, targets,epoch+self.iterationNo)
            self.optimizer.zero_grad()
            
            rgb_preds, pose_preds = self.model(imgs, headmap_vols)
            if self.debug:
                print("rgb_preds,pose_preds", rgb_preds, pose_preds)

           
            rgb_loss_value = F.cross_entropy(rgb_preds, targets)
            pose_loss_value = F.cross_entropy(pose_preds, targets)

            loss_value = pose_loss_value + rgb_loss_value
            loss_value.backward()
            # ====gradient clipping====
            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=40, norm_type=2.)
            # ==============================#

            self.optimizer.step()
            self.scheduler.step()

            # ===calculate top-1, top-5 acc
            rgb_scores = rgb_preds.detach().cpu().numpy()
            pose_scores = pose_preds.detach().cpu().numpy()
            output_np = rgb_scores + pose_scores
            targets_np = targets.cpu().numpy()
            #output_np, targets_np = rgb_preds.detach().cpu().numpy(), targets.cpu().numpy()
            top_k_acc = self.evalMetric.top_k_accuracy(output_np, targets_np, topk=(1, 5))
            #mean_class_accuracy = self.evalMetric.mean_class_accuracy(output_np, targets_np)
            for k, a in zip((1, 5), top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=loss_value.device)
            losses['loss'] = loss_value
            #losses['mean_class_acc'] = torch.tensor(mean_class_accuracy, device=loss_value.device)
            log_vars = self.parse_losses(losses)
            self.log_buffer.update(log_vars, imgs.shape[0])

            #print(log_vars)
            if iterationNo % self.log_interval == 0 and self.gpu_id == 0:
                #print("self.gpu_id == 0:",self.gpu_id)
                self.log_buffer.average(self.log_interval)
                # print(log_buffer.output)
                top1_acc, top5_acc, loss = self.log_buffer.output['top1_acc'], self.log_buffer.output[
                    'top5_acc'], self.log_buffer.output['loss']
                self.log_buffer.clear()
                # topK_acc=self.evalMetric.top_k_accuracy(preds ,np.array(labels),topk=(1,5))
                # ====log loss and acc====
                self.sWriter.add_scalar('Train/loss', loss, epoch * num_iterations + iterationNo)
                self.sWriter.add_scalar('Train/top1_acc', top1_acc, epoch * num_iterations + iterationNo)
                self.sWriter.add_scalar('Train/top5_acc', top5_acc, epoch * num_iterations + iterationNo)
                #self.sWriter.add_scalar('Train/mean_class_acc', mean_acc, epoch * num_iterations + iterationNo)
                #====get learning rate====#

                msg = f"{datetime.now()} Epoch: [{epoch}][{iterationNo}|{len(self.train_dataLoader)}] lr: {self.optimizer.param_groups[0]['lr']:.7f} loss: {loss:.5f} top1_acc: {top1_acc:.5f} top5_acc: {top5_acc:.5f}"
                #print(msg)
                self.logger.info(msg)

    def evaluate(self, epoch):
        self.model.eval()
        stepsNo = len(self.test_dataLoader)
        with torch.no_grad():
            num_iterations = len(self.test_dataLoader)
            model_preds = []
            gt_labels = []
            losses = dict()
            for iterationNo, (imgs, headmap_vols, targets) in enumerate(self.test_dataLoader):
                imgs = imgs.to(self.gpu_id)
                headmap_vols = headmap_vols.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                #rgb_preds,pose_preds,adaptive_score_fusion_logits = self.model(imgs, headmap_vols)
                rgb_preds,pose_preds = self.model(imgs, headmap_vols)
                rgb_loss_value = F.cross_entropy(rgb_preds, targets)
                pose_loss_value = F.cross_entropy(pose_preds, targets)
                 
                test_loss = pose_loss_value + rgb_loss_value
                #test_loss = pose_loss_value + rgb_loss_value+adaptive_loss_value
                rgb_scores = rgb_preds.detach().cpu().numpy()
                pose_scores = pose_preds.detach().cpu().numpy()
                preds_np = rgb_scores + pose_scores
               
                labels_np = targets.cpu().numpy()
                model_preds.extend(preds_np)
                gt_labels.extend(labels_np)
                #preds_np, labels_np = preds.detach().cpu().numpy(), targets.cpu().numpy()
                top_k_acc = self.evalMetric.top_k_accuracy(preds_np, labels_np, topk=(1, 5))
                # top_k_acc = self.evalMetric.top_k_accuracy(preds.detach().cpu().numpy(), targets.cpu().numpy(), topk=(1, 5))

                for k, a in zip((1, 5), top_k_acc):
                    losses[f'top{k}_acc'] = torch.tensor(
                        a, device=test_loss.device)
                losses['loss'] = test_loss
                log_vars = self.parse_losses(losses)
                self.test_log_buffer.update(log_vars, imgs.shape[0])


        gathered_labels, gathered_preds = self.collect_results_gpu(gt_labels, model_preds)
        if self.gpu_id == 0:
            self.test_log_buffer.average(num_iterations)
            top1_acc, top5_acc, loss = self.test_log_buffer.output['top1_acc'], self.test_log_buffer.output[
                'top5_acc'], self.test_log_buffer.output['loss']
            self.test_log_buffer.clear()
            print("num of samples: ", gathered_preds.shape)
            mean_class_accuracy = self.evalMetric.mean_class_accuracy(gathered_preds, gathered_labels)
            msg = f"Evaluating top_k_accuracy ...\n top1_acc: {top1_acc:.5f}\n top5_acc: {top5_acc:.5f}\n Evaluating mean_class_accuracy ...\n mean_acc :{mean_class_accuracy:.5f}"
            #print(msg)
            self.logger.info(msg)


            self.sWriter.add_scalar('Test/loss', loss, epoch)
            self.sWriter.add_scalar('Test/top1_acc', top1_acc, epoch)
            self.sWriter.add_scalar('Test/top5_acc', top5_acc, epoch)
            self.sWriter.add_scalar('Test/mean_class_acc', mean_class_accuracy, epoch)

            if top1_acc > self.best_top_1_acc:
                # ===remove the previous best_ck_pt and save the current one====#            msg = f"Metrics Sync top1_acc, top5_acc, loss: {top1_acc:.3f}, {top5_acc:.3f}, {loss:.3f}"
                self.best_top_1_acc = top1_acc

            self.logger.info(f"Best top-1 Acc is: {self.best_top_1_acc:.5f} at epoch: {epoch}")
            # self._save_checkpoint(epoch,save_best=True)
            # if os.path.exists(self.best_ck_pt_path):
            #     os.remove(self.best_ck_pt_path)

    def collect_results_gpu(self,gt_labels,model_preds):
        # ===collect data from all gpus
        world_size = torch.distributed.get_world_size()
        gathered_preds = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        # the first argument is the collected lists, the second argument is the data unique in each process
        torch.distributed.all_gather_object(gathered_preds, model_preds)
        torch.distributed.all_gather_object(gathered_labels, gt_labels)
        gathered_preds = np.asarray(gathered_preds)
        gathered_labels = np.asarray(gathered_labels)
        gathered_preds = gathered_preds.reshape(-1, gathered_labels.shape[-1])
        gathered_labels = gathered_labels.reshape(-1, gathered_labels.shape[-1])

        return gathered_labels,gathered_preds

    def do_inference(self):
        print("Inference")
        self.model.eval()
        with torch.no_grad():
            model_preds = []
            gt_labels = []
            for iterationNo, (imgs, headmap_vols, targets) in enumerate(self.test_dataLoader):
                imgs = imgs.to(self.gpu_id)
                headmap_vols = headmap_vols.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                rgb_preds,pose_preds = self.model(imgs, headmap_vols)
                rgb_loss_value = F.cross_entropy(rgb_preds, targets)
                pose_loss_value = F.cross_entropy(pose_preds, targets)

                test_loss = pose_loss_value + rgb_loss_value

                rgb_scores = rgb_preds.detach().cpu().numpy()
                pose_scores = pose_preds.detach().cpu().numpy()
                preds_np = rgb_scores + pose_scores
                labels_np = targets.cpu().numpy()
                model_preds.extend(preds_np)
                gt_labels.extend(labels_np)
        # #if self.gpu_id == 0:
        model_preds_ = np.asarray(model_preds)
        gt_labels_ = np.asarray(gt_labels)
        top_k_acc = self.evalMetric.top_k_accuracy(model_preds_, gt_labels_, topk=(1, 5))
        mean_class_accuracy = self.evalMetric.mean_class_accuracy(model_preds_, gt_labels_)
        top1_acc,top5_acc = top_k_acc[0],top_k_acc[1]
        msg = f"Evaluating top_k_accuracy ...\n top1_acc: {top1_acc:.5f}\n top5_acc: {top5_acc:.5f}\n Evaluating mean_class_accuracy ...\n mean_acc :{mean_class_accuracy:.5f}"
        print(msg)
        print("num of samples: ",model_preds_.shape)

        #===collect data from all gpus
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            # the first argument is the collected lists, the second argument is the data unique in each process
            torch.distributed.all_gather_object(gathered_preds, model_preds)
            torch.distributed.all_gather_object(gathered_labels, gt_labels)
            gathered_preds = np.asarray(gathered_preds)
            gathered_labels = np.asarray(gathered_labels)
            gathered_preds = gathered_preds.reshape(-1,31)
            gathered_labels = gathered_labels.reshape(-1,31)
            print("num of samples: ", gathered_preds.shape)
            print("num of samples(gathered_labels): ", gathered_labels.shape)
            top_k_acc = self.evalMetric.top_k_accuracy(gathered_preds, gathered_labels, topk=(1, 5))
            mean_class_accuracy = self.evalMetric.mean_class_accuracy(gathered_preds, gathered_labels)
            top1_acc, top5_acc = top_k_acc[0], top_k_acc[1]
            msg = f"Evaluating top_k_accuracy ...\n top1_acc: {top1_acc:.5f}\n top5_acc: {top5_acc:.5f}\n Evaluating mean_class_accuracy ...\n mean_acc :{mean_class_accuracy:.5f}"
            print(msg)
            print("num of samples: ", gathered_preds.shape)


