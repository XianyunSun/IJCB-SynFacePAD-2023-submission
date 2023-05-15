import numpy as np
import pandas as pd
import os
import wandb

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

from dataset import EqualSample, GridTest, get_test_set, data_concate
from utils import  AvgrageMeter, performances_cross_db
from model.model import OrthIDModel
from loss import CrossEntropyWithOPLoss


def main(train_list, test_list, id_list, args):
    train_dataset = EqualSample(train_list[0], train_list[1], id_list=id_list, input_shape=(224, 224), bi_flow=args.aug, match='min', jpg=0.6, eye=0.6)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, drop_last=True, shuffle=True)
    print('len of training set:', len(train_dataset))
    
    if test_list is not None:
        test_dataset = GridTest(test_list[0], test_list[1], input_shape=(224, 224), bi_flow=args.aug, eye=0)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, pin_memory=True, drop_last=True, shuffle=True)
        print('len of test set:', len(test_dataset))


    model = OrthIDModel(model_name=args.model_name, pretrained=False, num_classes=2)
    
    if torch.cuda.device_count()>0:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    cen_criterion = torch.nn.CrossEntropyLoss().cuda()
    contrastive_loss = torch.nn.CosineSimilarity().cuda()

    scaler = GradScaler()

    for epoch in range(args.train_epoch):
        torch.cuda.empty_cache()

        AUC_train, HTER_train, loss_total = train_epoch(model, train_loader, optimizer, cen_criterion, contrastive_loss, scaler)
        #AUC_train, HTER_train, loss_total = -1, -1, -1

        if test_list is not None:
            AUC_test, HTER_test = test_epoch(model, test_loader, prop=1)
        #AUC_M, HTER_M, AUC_C, HTER_C = -1, -1, -1, -1
        lr_scheduler.step()

        if args.log is not None:
            wandb.log({'train/AUC':AUC_train, 'train/HTER':HTER_train, 'train/loss':loss_total})
            if test_list is not None:
                wandb.log({'test/AUC':AUC_test, 'test/HTER':HTER_test})


    save_path = os.path.join(args.pth_path, args.model_name+'_id_'+str(args.seed)+'.pth')
    torch.save(model.state_dict(), save_path)
    print('model saved to:', save_path)


# ===================================================================================================================================
def train_epoch(model, train_loader, optimizer, cen_criterion, contrastive_loss, scaler):
    loss_total = AvgrageMeter()

    model.train()
    
    result_list = []
    gt_list = []
    for i, data in enumerate(tqdm(train_loader)):
    #for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        data = data_concate(data)
        input_data, input_data_aug = data['images'].cuda(), data['img_aug'].cuda()
        label, id_label = data['labels'].to(torch.int64).cuda(), data['id'].to(torch.int64).cuda()
        output = model(input_data, input_data_aug, None, None, id_label, train_flag=1)

        raw_scores = output['pred'].softmax(dim=1)[:, 1].cpu().data.numpy()

        # loss
        loss_pad = cen_criterion(output['pred'], label)
        loss_id = cen_criterion(output['pred_id'], id_label)
        loss_orth = torch.abs(contrastive_loss(output['feats_pad'], output['feats_id'])).mean()
        loss = loss_pad + loss_id*0.01 + loss_orth*10
        loss_total.update(loss.data, input_data.shape[0])

        #print('loss total=%.4f, pad loss=%.4f, id loss=%.4f, feats loss=%.4f' % (loss.data, loss_pad.data, loss_id.data, loss_orth.data))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        result_list.extend(raw_scores)
        gt_list.extend(data["labels"].data.numpy())
    
    result_stats = [np.mean(result_list), np.std(result_list)]
    raw_test_scores = ( result_list - result_stats[0]) / result_stats[1]
    AUC, _, _, HTER, _ = performances_cross_db(raw_test_scores, gt_list)
       
    return AUC, HTER, loss_total.avg


def test_epoch(model, data_loader, prop=1):
    model.eval()

    raw_test_scores, gt_labels = [], []
    stop_iter = int(len(data_loader)*prop)
    #stop_iter = 1
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if i==stop_iter: break

            #data = data_concate(data)
            labels = data['labels']
            output_list = []
            for i in range(len(data['images'])):
                raw, aug = data['images'][i].cuda(), data['img_aug'][i].cuda()
                output_single = model(raw, aug, None, None, id_labels=None, train_flag=0)
                #output_single = model(raw)
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

# ================================================================================================

if __name__ == "__main__":

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='training code of model2 of SynFacePAD-2023 submission')
    parser.add_argument("--model_name", default='resnet18', type=str, help="model backbone")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--train_epoch", default=1, type=int, help="train epochs")
    parser.add_argument("--aug", default='clahe', type=str, help='aug method for bi-flow')
    parser.add_argument("--train_batch_size", default=64, type=int, help="train batch size")
    parser.add_argument("--test_batch_size", default=64, type=int, help="test batch size")

    parser.add_argument("--train_dir", type=str, help="root dir of training set")
    parser.add_argument("--train_csv", type=str, help="dir of the label file of training set")
    parser.add_argument("--test_dir", type=str, help="root dir of test set, set to None to disable testing during training process")
    parser.add_argument("--test_csv", type=str, help="dir of the label file of test set")

    parser.add_argument("--pth_path", default='./pth', type=str, help="dir for saving trained model")
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--log", default=None, type=str, help="name for wandb logging, set to None to disable wandb" )

    args = parser.parse_args()
    set_seed(seed=args.seed)
    
    train_list = get_test_set([args.train_dir], [args.train_csv], prop=1)

    # get id list
    train_csv_df = pd.read_csv(args.train_csv)
    id_list = list(train_csv_df['id'])

    if args.test_dir == 'None': test_list = None
    else: test_list = get_test_set([args.test_dir], [args.test_csv], prop=1)

    if args.log is not None:
        wandb.init(project='SynFacePAD-2023-BUCEA team', name=args.log, reinit=True)
    
    main(train_list, test_list, id_list, args)
