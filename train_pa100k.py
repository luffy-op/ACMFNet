import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine_fcl2lmask import valid_trainer, batch_trainer
from config_pa100k import argument_parser #配置文件
from dataset.CleanPA100k import AttrDataset #数据集类
from dataset.AttrDataset import get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block_fcl2l import FeatClassifier_fcl2lmask, BaseClassifier
from myclassifier.swin_fc_cls import MyMultilabelClassifier_SwinS
from models.resnet import resnet50
from models.model import FrozenBatchNorm2d
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, seedForExp

from models.visual_model import Image_encoder_ModifiedResNet
from torchvision.models import swin_transformer
# from torchsummary import summary
# from timm.models.swin_transformer import swin_small_patch4_window7_224
# from models.swin_transformer_v2 import SwinTransformerV2
# set_seed(605)

pth_filename = 'ckpt_swins_fcl2lnomask_mask_A_seed213205_max.pth'
metric_filename = 'metric_swins_fcl2lnomask_mask_A_seed213205_log.pkl'

def main(args):
    seedForExp(args) # 随机种子
    visenv_name = args.dataset #数据集名
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt') #日志
    # save_model_path = os.path.join(model_dir, 'ckpt_pretrain_max.pth')
    save_model_path = model_dir

    if args.redirector: #输出重定向
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))
    
    print('-' * 60)
    print(f'use GPU{args.gpu} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')
    torch.cuda.set_device(args.gpu)
    train_tsfm, valid_tsfm = get_transform(args) #transform
    print(train_tsfm)

    # train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_set = AttrDataset(split=args.train_split, transform=train_tsfm, testing=False, known_labels=100)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    # valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_set = AttrDataset(split=args.valid_split, transform=valid_tsfm, testing=True, known_labels=0)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    swin_s_model = swin_transformer.swin_s(pretrained=True)
    backbone = torch.nn.Sequential(*(list(swin_s_model.children())[:-3]))

    # backbone = Image_encoder_ModifiedResNet([3,4,6,3],768,8)
    # backbone = resnet50()
    classifier = BaseClassifier(nattr=train_set.attr_num, args=args)
    classifier_fc = MyMultilabelClassifier_SwinS(args.num_att)
    model = FeatClassifier_fcl2lmask(backbone, classifier, classifier_fc)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight=sample_weight,ratio=args.ratio, pos_weight=args.pos_weight)

    param_groups = [{'params': model.backbone.parameters(), 'lr': args.lr_ft},
                    {'params': model.classifier.parameters(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 args = args)

    print(f'{visenv_name},  best_ma : {best_metric[0]} in epoch{epoch[0]}')
    print(f'{visenv_name},  best_acc: {best_metric[1]} in epoch{epoch[1]}')
    print(f'{visenv_name},  best_f1 : {best_metric[2]} in epoch{epoch[2]}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, args):
    maximum1, maximum2, maximum3 = float(-np.inf), float(-np.inf),float(-np.inf)
    best_epoch1, best_epoch2, best_epoch3 = 0,0,0

    result_list = defaultdict()

    # 加载backbone权重
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # checkpoint = torch.load("checkpoints/PLIP_RN50.pth.tar", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # model.backbone.load_state_dict(checkpoint["ImgEncoder_state_dict"], strict=False)

    for i in range(epoch):
        
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            args = args
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric1 = valid_result.ma
        cur_metric2 = valid_result.instance_acc
        cur_metric3 = valid_result.instance_f1
        if cur_metric1 > maximum1:
            maximum1 = cur_metric1
            best_epoch1 = i
            save_ckpt(model, os.path.join(path, pth_filename), i, maximum1)
            # save_ckpt(model, '/home/zhexuan_wh/model/m.pth', i, maximum1)
            # torch.save({'pt': valid_probs, 'gt':valid_gt}, '/home/zhexuan_wh/model/result.pkl')
        
        if cur_metric2 > maximum2:
            maximum2 = cur_metric2
            best_epoch2 = i
        if cur_metric3 > maximum3:
            maximum3 = cur_metric3
            best_epoch3 = i
        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), metric_filename))

    return (maximum1,maximum2,maximum3), (best_epoch1,best_epoch2,best_epoch3)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
