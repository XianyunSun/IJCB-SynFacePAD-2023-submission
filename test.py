import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import argparse

from dataset import GridTest, get_test_set
from utils import performances_cross_db
from model.model import OrthIDModel, OrthBiModel

def main(test_list, model_list, args):

    test_dataset = GridTest(test_list[0][0], test_list[0][1], input_shape=(224, 224), bi_flow='clahe', eye=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pred_list = []
    th_list = []
    print(f'there are {len(model_list)} models')
    for i in range(len(model_list)):
        torch.cuda.empty_cache()
        print('loading model from:', model_list[i])
        if 'id' in model_list[i]:
            model = OrthIDModel(model_name='resnet18', pretrained=False, num_classes=2)
        elif 'res' in model_list[i]:
            model = OrthBiModel(model_name='resnet18', pretrained=False, num_classes=2)

        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
        model.load_state_dict(torch.load(model_list[i])) 

        AUC, HTER, output_dict = test_epoch(model, test_loader)
        print('test result from model %d: AUC=%.4f, HTER=%.4f' % (i+1, AUC, HTER))

        if len(pred_list)==0:
            pred_list.append(pd.DataFrame({'image_path':output_dict['image_path'], 'true_label':output_dict['true_label']}))
        
        pred_dict = {'image_path':output_dict['image_path'], 'prediction_score':output_dict['prediction_score'], 'prediction_label':output_dict['prediction_label']}
        pred_list.append(pd.DataFrame(pred_dict, index=None))
        th_list.append(output_dict['th'])


    # save results
    if args.vote:
        pred_df = vote(pred_list, th_list)
    else:
        pred_df = ensembel(pred_list)
    
    pred_df.to_csv(os.path.join(args.output_file, 'pred_results.csv'), index=False)

# =============================================================================================================

def test_epoch(model, data_loader):
    torch.cuda.empty_cache()
    model.eval()

    raw_test_scores, gt_labels = [], []
    img_path_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            labels = data['labels']
            img_path = data['image_path']
            output_list = []
            for i in range(len(data['images'])):
                raw, aug = data['images'][i].cuda(), data['img_aug'][i].cuda()
                output_single = model(raw, aug, None, None, train_flag=0)
                output_single = output_single.softmax(dim=1)[:, 1].cpu().data.numpy()
                output_list.append(output_single)

            raw_scores = np.asarray(output_list).mean(axis=0)
            raw_test_scores.extend(raw_scores)
            gt_labels.extend(labels.data.numpy())
            img_path_list.extend(img_path)

        raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
        raw_test_scores = ( raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

        AUC, _, _, HTER, th = performances_cross_db(raw_test_scores, gt_labels)

        pred_labels = list(map(int, (raw_test_scores>th)))
        output_dict = {'image_path':img_path_list, 'true_label':gt_labels, 'prediction_score':raw_test_scores, 'prediction_label': pred_labels, 'th':th}

        return AUC, HTER, output_dict


# =============================================================================================================
def ensembel(pred_list):
    if len(pred_list)==1: return pred_list[0]
    else:
        d0 = pred_list[0]
        gt_label = np.array(d0['true_label'])
        for i in range(1,len(pred_list)):
            d = pred_list[i]
            d0 = d0.merge(d.rename(columns={'prediction_score':'prediction_score'+str(i)}), on='image_path', how='inner')

    # calculate average score
    scores = []
    for i in range(len(pred_list)-1):
        scores.append(np.array(d0['prediction_score'+str(i+1)]))
    scores = np.stack(scores, axis=0).mean(axis=0)

    AUC, _, _, HTER, th = performances_cross_db(scores, gt_label)
    print('average test result: AUC=%.4f, HTER=%.4f' % (AUC, HTER))

    pred_labels = list(map(int, (scores>th)))

    d0['prediction_score'] = scores
    d0['prediction_label'] = pred_labels

    return d0

def vote(pred_list, th_list):
    d0 = pred_list[0]
    gt_label = np.array(d0['true_label'])
    for i in range(1,len(pred_list)):
        d = pred_list[i]
        th = th_list[i-1]
        d_join = d0.merge(d, on='image_path', how='inner')
        pred_scores = np.array(d_join['prediction_score'], dtype=float)
        confidence = np.abs(pred_scores-th)
        d0['prediction_label_'+str(i)] = list(d_join['prediction_label'])
        d0['confidence_'+str(i)] = confidence

    pred_col, confidence_col = [], []
    for i in range(1,len(pred_list)):
        pred_col.append('prediction_label_'+str(i))
        confidence_col.append('confidence_'+str(i))

    pred_label_loc = [ l.replace('confidence_', 'prediction_label_') for l in list(d0[confidence_col].idxmax(axis=1))]
    pred_label = []
    for i in range(len(gt_label)):
        pred_label.append(d0[pred_label_loc[i]][i])

    d0['prediction_label'] = pred_label

    accurancy = np.count_nonzero(pred_label==gt_label) / float(len(gt_label))
    print('test accurancy after voting: %.4f' % accurancy)

    return d0


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


# =========================================================================================================
if __name__ == "__main__":

    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='testing code of SynFacePAD-2023 submission')
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--batch_size", default=64, type=int, help="train batch size")
    parser.add_argument("--test_data_dir", type=str, help="root path of test dataset")
    parser.add_argument("--test_csv", type=str, help="csv file for testing")
    parser.add_argument("--model_path", default='./pth', type=str, help="test model weight path")
    parser.add_argument("--output_file", default='./result', type=str, help="direction of result csv")
    parser.add_argument("--seed", default=181, type=int, help="random seed")
    parser.add_argument("--vote", default=False, type=bool, help="Wether to use the vote strategy")

    args = parser.parse_args()
    set_seed(seed=args.seed)

    args.test_data_dir = r'/sdata/xianyun.sun/MSU_crop/image'
    args.test_csv = r'/sdata/xianyun.sun/MSU_crop/labels.csv'

    model_list = os.listdir(args.model_path)
    model_list = [os.path.join(args.model_path, m) for m in model_list]
    model_list = ['./pth/OrthBi_resnet_id_2.pth', './pth/OrthBi_resnet_jpg_eye_crop_2.pth']
    test_list = get_test_set([args.test_data_dir], [args.test_csv], prop=1)

    main(test_list=[test_list], model_list=model_list, args=args)
