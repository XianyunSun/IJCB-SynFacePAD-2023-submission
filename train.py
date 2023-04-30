import numpy as np
import pandas as pd
import os
import wandb
import copy
#import logging
from tqdm import tqdm
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append('SynthASpoof/SynthASpoof-main/')

from dataset import EqualSample, GridTest, get_test_set
from utils import  AvgrageMeter, performances_cross_db
from model.model import pixelBCELoss, ContrastiveLoss, OrthFcModel, OrthBiModel
from loss import OrthogonalProjectionLoss, CrossEntropyWithOPLoss


def main(train_list, args):

    train_dataset = EqualSample(train_list[0], train_list[1], input_shape=(224, 224), train=True, bi_flow=args.aug, match='min')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True)
    print('len of training set:', len(train_dataset))

    #model = OrthFcModel(model_name=args.model_name, pretrained=True, num_classes=2, iqa=True)
    model = OrthBiModel(model_name=args.model_name, pretrained=False, num_classes=2)

    if torch.cuda.device_count()>0:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    cen_criterion_tune = torch.nn.CrossEntropyLoss().cuda()
    #cen_criterion = torch.nn.CrossEntropyLoss().cuda()
    cen_criterion = CrossEntropyWithOPLoss()
    contrastive_loss = torch.nn.CosineSimilarity().cuda()

    scaler = GradScaler()

    # ========================================================= train
    for epoch in range(args.train_epoch):

        AUC_train, HTER_train, loss_total = train_epoch(model, train_loader, optimizer, cen_criterion, contrastive_loss, scaler)

        if args.log is not None:
            wandb.log({'train/AUC':AUC_train, 'train/HTER':HTER_train, 'train/loss':loss_total})

        lr_scheduler.step()
    
    # ========================================================= fine tune with test
    for p in model.module.feature_extractor.parameters():
        p.requires_grad = False
    for p in model.module.feature_extractor_aug.parameters():
        p.requires_grad = False
    #for p in model.module.backbone.parameters():
        #p.requires_grad = False
    #for p in model.module.IQA.parameters():
        #p.requires_grad = False
    
    for epoch in range(args.fine_tune_epoch):
        AUC_tune, HTER_tune, loss_tune = fine_tune_epoch(model, train_loader, optimizer, cen_criterion_tune, optimizer.param_groups[0]['lr'], scaler)

        if args.log is not None:
            wandb.log({'train/AUC':AUC_tune, 'train/HTER':HTER_tune, 'train/loss':loss_tune})
        #print('tune AUC: %.4f' % (AUC_tune))
            
        lr_scheduler.step()

    # ========================================================= save final model
    if not os.path.exists(args.pth_path):
        os.makedirs(args.pth_path)

    save_path = os.path.join(args.pth_path, args.model_name+'_'+str(args.seed)+'.pth')
    torch.save(model.state_dict(), save_path)
    print('model saved to:', save_path)


# ==========================================================================================================================  

def train_epoch(model, train_loader, optimizer, cen_criterion, contrastive_loss, scaler):
    loss_total = AvgrageMeter()
    model.train()
    
    result_list = []
    gt_list = []

    torch.cuda.empty_cache()
    for i, data in enumerate(tqdm(train_loader)):
    #for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        #img_bona, label_bona = data['img_bona'].cuda(), data['label_bona'].cuda()
        #img_att, label_att = data['img_att'].cuda(), data['label_att'].cuda()
        img_bona, img_bona_aug, label_bona = data['img_bona'].cuda(), data['img_bona_aug'].cuda(), data['label_bona'].cuda()
        img_att, img_att_aug, label_att = data['img_att'].cuda(), data['img_att_aug'].cuda(), data['label_att'].cuda()
        output = model(img_bona, img_bona_aug, img_att, img_att_aug, train_flag=True)
        #output = model(img_bona, img_att, train_flag=True)
        
        # classification loss
        pred = torch.cat((output['pred_bona'], output['pred_att']), dim=0)
        feats_bona = torch.cat((output['feats_bona_unique'], output['feats_bona_common']), dim=1)
        feats_att = torch.cat((output['feats_att_unique'], output['feats_att_common']), dim=1)
        feats = torch.cat((feats_bona, feats_att), dim=0)
        label = torch.cat((label_bona, label_att), dim=0)
        raw_scores = pred.softmax(dim=1)[:, 1].cpu().data.numpy()
        #binary_loss = cen_criterion(pred, label)
        binary_loss = cen_criterion(feats, pred, label)

        feats_bona_loss = torch.abs(contrastive_loss(output['feats_bona_unique'], output['feats_bona_common']).mean())
        feats_att_loss = torch.abs(contrastive_loss(output['feats_att_unique'], output['feats_att_common']).mean())

        # loss
        loss = binary_loss + feats_bona_loss + feats_att_loss
        #print('bi loss= %.4f, bona loss= %.4f, att loss= %.4f, total loss=%.4f' 
        #       % (binary_loss.item(), feats_bona_loss.item(), feats_att_loss.item(), loss.item()))

        loss_total.update(loss.data, img_bona.shape[0]*2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        result_list.extend(raw_scores)
        gt_list.extend(label.cpu().data.numpy())
    
    result_stats = [np.mean(result_list), np.std(result_list)]
    raw_test_scores = ( result_list - result_stats[0]) / result_stats[1]
    AUC, _, _, HTER, _ = performances_cross_db(raw_test_scores, gt_list)
       
    return AUC, HTER, loss_total.avg

def fine_tune_epoch(model, data_loader, optimizer, cen_criterion, current_lr, scaler, log=False):
    loss_total = AvgrageMeter()
    model.train()
    
    result_list = []
    gt_list = []  

    torch.cuda.empty_cache()
    for i, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        
        img_bona, img_bona_aug, label_bona = data['img_bona'].cuda(), data['img_bona_aug'].cuda(), data['label_bona'].cuda()
        img_att, img_att_aug, label_att = data['img_att'].cuda(), data['img_att_aug'].cuda(), data['label_att'].cuda()
        #img_bona, label_bona = data['img_bona'].cuda(), data['label_bona'].cuda()
        #img_att, label_att = data['img_att'].cuda(), data['label_att'].cuda()
        img = torch.cat((img_bona, img_att), dim=0)
        img_aug = torch.cat((img_bona_aug, img_att_aug), dim=0)
        pred = model(img, img_aug, None, None, train_flag=False)
        #pred = model(img, None, train_flag=False)
        
        # classification loss
        label = torch.cat((label_bona, label_att), dim=0)
        raw_scores = pred.softmax(dim=1)[:, 1].cpu().data.numpy()
        loss = cen_criterion(pred, label)

        #print('bi loss= %.4f' % (loss.item()))

        loss_total.update(loss.data, img_bona.shape[0]*2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        result_list.extend(raw_scores)
        gt_list.extend(label.cpu().data.numpy())
    
    result_stats = [np.mean(result_list), np.std(result_list)]
    raw_test_scores = ( result_list - result_stats[0]) / result_stats[1]
    AUC, _, _, HTER, _ = performances_cross_db(raw_test_scores, gt_list)
       
    # save model
    #torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f'{epoch}.pth'))

    return AUC, HTER, loss_total.avg

def test_epoch(model, data_loader, prop=1):
    model.eval()

    raw_test_scores, gt_labels = [], []
    stop_iter = int(len(data_loader)*prop)
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if i==stop_iter: break

            labels = data['labels']
            output_list = []
            for i in range(len(data['images'])):
                raw, aug= data['images'][i].cuda(), data['img_aug'][i].cuda()
                #raw, img_pathes = data['images'][i].cuda(), data["image_path"][i]
                output_single = model(raw, aug, None, None, train_flag=0)
                #output_single = model(raw, None, train_flag=0)
                output_single = output_single.softmax(dim=1)[:, 1].cpu().data.numpy()
                output_list.append(output_single)
            
            raw_scores = np.asarray(output_list).mean(axis=0)
            #raw_scores = output.softmax(dim=1)[:, 1].cpu().data.numpy()
            raw_test_scores.extend(raw_scores)
            gt_labels.extend(labels.data.numpy())

        raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
        raw_test_scores = ( raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

        AUC, _, _, HTER, _ = performances_cross_db(raw_test_scores, gt_labels)

    return AUC, HTER

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    #torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def data_concate(data):
    img_bona, img_att = data['img_bona'], data['img_att']
    img_bona_aug, img_att_aug = data['img_bona_aug'], data['img_att_aug']
    label_bona, label_att = data['label_bona'], data['label_att']
    img_bona_path, img_att_path = data['img_bona_path'], data['img_att_path']
    img_path = img_bona_path + img_att_path
    img = torch.cat((img_bona, img_att), dim=0)
    img_aug = torch.cat((img_bona_aug, img_att_aug), dim=0)
    label = torch.cat((label_bona, label_att), dim=0)

    return {'images':img, 'img_aug':img_aug, 'labels':label, 'image_path':img_path}

def train_val_split(img_list, label_list, test_prop=0.2, random_state=0):
    label_bona_idx = np.argwhere(label_list=='bonafide').squeeze()
    label_att_idx = np.argwhere(label_list!='bonafide').squeeze()

    img_bona = np.asarray(img_list)[label_bona_idx]
    img_att = np.asarray(img_list)[label_att_idx]
    label_bona = np.asarray(label_list)[label_bona_idx]
    label_att = np.asarray(label_list)[label_att_idx]

    train_img_bona, val_img_bona, train_label_bona, val_label_bona = train_test_split(img_bona, label_bona, test_size=test_prop, random_state=random_state)
    train_img_att, val_img_att, train_label_att, val_label_att = train_test_split(img_att, label_att, test_size=test_prop, random_state=random_state)

    train_img = list(train_img_bona)+list(train_img_att)
    train_label = list(train_label_bona)+list(train_label_att)
    val_img = list(val_img_bona)+list(val_img_att)
    val_label = list(val_label_bona)+list(val_label_att)

    return train_img, val_img, train_label, val_label


# ================================================================================================

if __name__ == "__main__":

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='training code of SynFacePAD-2023 submission')
    parser.add_argument("--model_name", default='resnet18', type=str, help="model backbone")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--train_epoch", default=1, type=int, help="train epochs")
    parser.add_argument("--fine_tune_epoch", default=1, type=int, help="fine tune epochs")
    parser.add_argument("--aug", default='clahe', type=str, help='aug method for bi-flow')
    parser.add_argument("--batch_size", default=64, type=int, help="train batch size")

    parser.add_argument("--train_dir", type=str, help="root dir of training set")
    parser.add_argument("--train_csv", type=str, help="dir of the label file of training set")

    parser.add_argument("--pth_path", default='./pth', type=str, help="dir for saving trained model")
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--log", default=None, type=bool, help="name for wandb logging, set to None to disable wandb" )

    args = parser.parse_args()
    set_seed(seed=args.seed)

    train_list = get_test_set([args.train_dir], [args.train_csv], prop=1)

    if args.log is not None:
        wandb.init(project='SynFacePAD-2023-test', name=args.log, reinit=True)
    
    main(train_list, args)
