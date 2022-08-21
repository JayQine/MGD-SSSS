from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from torchvision import models
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from criterion import CriterionCosineSimilarity

from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

# os.environ['MASTER_PORT'] = '29005'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False


'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader

    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    # cosine
    criterion_cos = CriterionCosineSimilarity()


    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # load pretrained model for teacher
    model_dict1 = torch.load(config.pretrained_tea1)['model']
    model_dict1 = {k: v for k, v in model_dict1.items() if 'branch1' in k}
    model_dict1 = {k.replace('branch1', 'branch2'): v for k, v in model_dict1.items()}
    model.load_state_dict(model_dict1, strict=False)

    model_dict2 = torch.load(config.pretrained_tea2)['model']
    model_dict2 = {k: v for k, v in model_dict2.items() if 'branch1' in k}
    model_dict2 = {k.replace('branch1', 'branch3'): v for k, v in model_dict2.items()}
    model.load_state_dict(model_dict2, strict=False)

    # set the lr
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define two optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_3 = []
    params_list_3 = group_weight(params_list_3, model.branch3.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch3.business_layer:
        params_list_3 = group_weight(params_list_3, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_3 = torch.optim.SGD(params_list_3,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)


    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_loss_sup_3 = 0
        sum_consist_loss = 0
        sum_sem_loss = 0
        sum_cont_loss = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                _, _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
                _, _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
                _, _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()
                # Estimate the pseudo-label with branch#3 & supervise branch#1
                _, _, logits_u0_tea_3 = model(unsup_imgs_0, step=3)
                _, _, logits_u1_tea_3 = model(unsup_imgs_1, step=3)
                logits_u0_tea_3 = logits_u0_tea_3.detach()
                logits_u1_tea_3 = logits_u1_tea_3.detach()


            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()
            logits_cons_tea_3 = logits_u0_tea_3 * (1 - batch_mix_masks) + logits_u1_tea_3 * batch_mix_masks
            _, ps_label_3 = torch.max(logits_cons_tea_3, dim=1)
            ps_label_3 = ps_label_3.long()

            # Get student#1 prediction for mixed image
            backbone_feat1, aspp_feat1, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            # Get teacher#2 prediction for mixed image
            backbone_feat2, aspp_feat2, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)
            # Get teacher#3 prediction for mixed image
            backbone_feat3, aspp_feat3, logits_cons_stu_3 = model(unsup_imgs_mixed, step=3)
            

            consist_loss1 = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
            dist.all_reduce(consist_loss1, dist.ReduceOp.SUM)
            consist_loss1 = consist_loss1 / engine.world_size
            
            
            consist_loss2 = criterion(logits_cons_stu_1, ps_label_3) + criterion(logits_cons_stu_3, ps_label_1)
            dist.all_reduce(consist_loss2, dist.ReduceOp.SUM)
            consist_loss2 = consist_loss2 / engine.world_size
            

            consist_loss_u = consist_loss1 + consist_loss2


            # image-level semantic-sensitive loss
            cls_logit1 = F.adaptive_avg_pool2d(logits_cons_stu_1, (1, 1))
            cls_logit3 = F.adaptive_avg_pool2d(logits_cons_stu_3, (1, 1))
            sem_sens_loss = torch.sum(torch.abs(cls_logit1 - cls_logit3))
            dist.all_reduce(sem_sens_loss, dist.ReduceOp.SUM)
            sem_sens_loss = sem_sens_loss / engine.world_size
            sem_sens_loss = sem_sens_loss * config.lambda_1

            # region-level content-aware loss
            q1 = F.adaptive_avg_pool2d(aspp_feat1, (7, 7))
            q2 = F.adaptive_avg_pool2d(aspp_feat2, (7, 7))
            q1 = q1.flatten(start_dim=2)
            q2 = q2.flatten(start_dim=2)

            self_sim_matrix_1 = criterion_cos(q1, q1)
            self_sim_matrix_2 = criterion_cos(q2, q2)

            content_loss = F.mse_loss(self_sim_matrix_1, self_sim_matrix_2)
            dist.all_reduce(content_loss, dist.ReduceOp.SUM)
            content_loss = content_loss / engine.world_size
            content_loss = content_loss * config.lambda_2
            

            # supervised loss on both models
            backbone_feat_l, _, sup_pred_l = model(imgs, step=1)
            backbone_feat_r, _, sup_pred_r = model(imgs, step=2)
            backbone_feat_3, _, sup_pred_3 = model(imgs, step=3)

            _, max_l = torch.max(sup_pred_l, dim=1)
            _, max_r = torch.max(sup_pred_r, dim=1)
            _, max_3 = torch.max(sup_pred_3, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()
            max_3 = max_3.long()

            consist_loss_s1 = criterion(sup_pred_l, max_r) + criterion(sup_pred_r, max_l)
            dist.all_reduce(consist_loss_s1, dist.ReduceOp.SUM)
            consist_loss_s1 = consist_loss_s1 / engine.world_size
            

            consist_loss_s2 = criterion(sup_pred_l, max_3) + criterion(sup_pred_3, max_l) 
            dist.all_reduce(consist_loss_s2, dist.ReduceOp.SUM)
            consist_loss_s2 = consist_loss_s2 / engine.world_size
            
            
            consist_loss_s = consist_loss_s1 + consist_loss_s2
            consist_loss = consist_loss_u + consist_loss_s


            loss_sup = criterion(sup_pred_l, gts)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / engine.world_size

            loss_sup_3 = criterion(sup_pred_3, gts)
            dist.all_reduce(loss_sup_3, dist.ReduceOp.SUM)
            loss_sup_3 = loss_sup_3 / engine.world_size

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # print(len(optimizer.param_groups))
            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr


            loss = loss_sup + loss_sup_r + loss_sup_3 + consist_loss + sem_sens_loss + content_loss
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_sup_3=%.2f' % loss_sup_3.item() \
                        + ' loss_consist=%.4f' % consist_loss.item() \
                        + ' loss_sem=%.4f' % sem_sens_loss.item() \
                        + ' loss_cont=%.4f' % content_loss.item() \
                        

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_loss_sup_3 += loss_sup_3.item()
            sum_consist_loss += consist_loss.item()
            sum_sem_loss += sem_sens_loss.item()
            sum_cont_loss += content_loss.item()
            
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
            logger.add_scalar('train_loss_consist', sum_consist_loss / len(pbar), epoch)
            logger.add_scalar('train_loss_sem', sum_sem_loss / len(pbar), epoch)
            logger.add_scalar('train_loss_cont', sum_cont_loss / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss Consist', value=sum_consist_loss / len(pbar))
            run.log(name='Supervised Training Loss Sem', value=sum_sem_loss / len(pbar))
            run.log(name='Supervised Training Loss Cont', value=sum_cont_loss / len(pbar))

        if (epoch > config.nepochs -15) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)