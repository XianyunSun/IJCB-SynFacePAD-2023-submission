import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
sys.path.append('/sdata/xianyun.sun/SynthASpoof_submit')
sys.path.append('/sdata/xianyun.sun/SynthASpoof_submit/model')
from head import AdaFace

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def pixelBCELoss(input, label):
    labels = label.repeat(input.shape[1], 1).permute(1,0).to(torch.float)
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_func(input, labels)
    loss = loss.mean()
    return loss

def smoothL1loss(input, label):
    labels = label.repeat(input.shape[1], 1).permute(1,0).to(torch.float)
    loss_func = nn.SmoothL1Loss(reduction='mean')
    loss = loss_func(input, labels)
    return loss
    
def generate_head(in_dim, channels):
    channels.insert(0, in_dim)
    layers = nn.Sequential()
    for i in range(len(channels)-1):
        layers.add_module('1*1conv'+str(i), nn.Conv2d(channels[i], channels[i+1], kernel_size=1))
        layers.add_module('bn'+str(i), nn.BatchNorm2d(channels[i+1]))
        if i!=len(channels)-2: layers.add_module('ReLU'+str(i), nn.ReLU())
    return layers

def generate_head_fc(in_dim, channels):
    channels.insert(0, in_dim)
    layers = nn.Sequential()
    for i in range(len(channels)-1):
        layers.add_module('fc'+str(i), nn.Linear(channels[i], channels[i+1]))
        if i!=len(channels)-2: layers.add_module('ReLU'+str(i), nn.ReLU())
    return layers

def generate_conv(in_dim, layers=3):
    conv = nn.Sequential()
    for i in range(layers):
        conv.add_module('conv'+str(i), nn.Conv2d(in_dim, in_dim, kernel_size=3, padding='same'))
        conv.add_module('bn'+str(i), nn.BatchNorm2d(in_dim))
        if i!=layers-1: conv.add_module('ReLU'+str(i), nn.ReLU())
    
    return conv

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label=0 for positive pair
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=2):
        super(Classifier, self).__init__()

        self.classifier_layer = nn.Linear(in_channels, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if(norm_flag): # norm classification weight
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            output = self.classifier_layer(input)
        else:
            output = self.classifier_layer(input)
        return output

class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Residual, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x, relu=True):
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out += identity
        if relu:
            out = self.relu(out)
        return out


# ================================================================================== models       
class OrthBiModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False, num_classes=2, channels=[512, 256]):

        super(OrthBiModel, self).__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_extractor = nn.Sequential(*nn.ModuleList(backbone.children()))
        backbone_aug = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_extractor_aug = nn.Sequential(*nn.ModuleList(backbone_aug.children()))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.in_channels = backbone.feature_info[-1]['num_chs']

        self.head_bona = generate_head_fc(self.in_channels*2, channels)
        self.head_att = generate_head_fc(self.in_channels*2, channels)
        self.head_common = generate_head_fc(self.in_channels*2, channels)

        head_in_dim = channels[-1]
        self.head2_bona = generate_head_fc(head_in_dim*2, [head_in_dim, num_classes])
        self.head2_att = generate_head_fc(head_in_dim*2, [head_in_dim, num_classes])
        self.head_classification = generate_head_fc(head_in_dim, [head_in_dim, num_classes])

    def forward_single(self, img, img_aug):
        f = self.feature_extractor(img)
        f_feat = self.avgpool(f).squeeze()
        f_aug = self.feature_extractor_aug(img_aug)
        f1_feat = self.avgpool(f_aug).squeeze()

        feats = torch.concat((f_feat, f1_feat), dim=1)
        return feats
    
    def forward(self, img_bona, img_bona_aug, img_att, img_att_aug, train_flag=True):
        map_bona = self.forward_single(img_bona, img_bona_aug)

        feats_bona_unique = self.head_bona(map_bona)
        feats_bona_common = self.head_common(map_bona)

        if not train_flag:  # testing phase
            pred = self.head_classification(feats_bona_common)
            return pred

        else:   # training phase
            map_att = self.forward_single(img_att, img_att_aug)

            feats_att_unique = self.head_att(map_att)
            feats_att_common = self.head_common(map_att)

            feats_bona = torch.cat((feats_bona_unique, feats_bona_common), dim=1)
            feats_att = torch.cat((feats_att_unique, feats_att_common), dim=1)
            pred_bona = self.head2_bona(feats_bona)
            pred_att = self.head2_att(feats_att)

            return {'pred_bona':pred_bona, 'pred_att':pred_att, 
                    'feats_bona_unique':feats_bona_unique, 'feats_bona_common':feats_bona_common,
                    'feats_att_unique':feats_att_unique, 'feats_att_common':feats_att_common}

class OrthIDModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False, num_classes=2, num_id=554997):

        super(OrthIDModel, self).__init__()
        if ('resnet' in model_name) or ('densenet' in model_name):
            backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            self.feature_extractor = nn.Sequential(*nn.ModuleList(backbone.children()))
            backbone_aug = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            self.feature_extractor_aug = nn.Sequential(*nn.ModuleList(backbone_aug.children()))

            self.in_channels = backbone.feature_info[-1]['num_chs']

        elif 'coatnet' in model_name or 'convnext' in model_name:
            backbone = timm.create_model(model_name, pretrained=pretrained)
            backbone_aug = timm.create_model(model_name, pretrained=pretrained)
            feature, feature_aug = [], []
            for bk in backbone.children(): feature.append(bk)
            for bk in backbone_aug.children(): feature_aug.append(bk)
            self.feature_extractor = nn.Sequential(*nn.ModuleList(feature[:-1]))
            self.feature_extractor_aug = nn.Sequential(*nn.ModuleList(feature_aug[:-1]))
            
            inp = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = self.feature_extractor(inp)
                self.in_channels = out.shape[1]
                print(out.shape)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.head_pad = generate_head_fc(self.in_channels*2, [self.in_channels, self.in_channels])
        self.head_id = generate_head_fc(self.in_channels*2, [self.in_channels, self.in_channels])
        
        self.head_classification = generate_head_fc(self.in_channels, [self.in_channels, num_classes])
        self.head_id_classification = AdaFace(embedding_size=self.in_channels, classnum=num_id, s=64)

    
    def forward(self, img, img_aug, img_att=None, img_att_aug=None, id_labels=None, train_flag=True):
        
        f = self.feature_extractor(img)
        f_feat = self.pool(f).squeeze()
        f_aug = self.feature_extractor_aug(img_aug)
        f1_feat = self.pool(f_aug).squeeze()

        feats = torch.concat((f_feat, f1_feat), dim=1)
        feats_pad = self.head_pad(feats)
        feats_id = self.head_id(feats)

        pred = self.head_classification(feats_pad)

        if not train_flag:  return pred

        else:
            norm = torch.norm(feats_id, 2, 1, True)
            feats_id_norm = torch.div(feats_id, norm)
            pred_id = self.head_id_classification(feats_id_norm, norm, id_labels)
            
            return {'pred':pred, 'pred_id':pred_id, 
                    'feats_pad':feats_pad, 'feats_id':feats_id}


# ===================================================================================

def _test():
    import torch
    labels = torch.tensor([0, 1, 1, 0])
    image_x = torch.randn(4, 3, 224, 224)

    model = OrthBiModel(model_name='resnet18', iqa=True)  
    #print(model.head_common.parameters)
    feats = model(image_x, image_x, image_x, image_x)
    #feats_bona = torch.cat((feats['feats_bona_unique'], feats['feats_bona_common']), dim=1)
    #feats_att = torch.cat((feats['feats_att_unique'], feats['feats_att_common']), dim=1)
    #feats = torch.cat((feats_bona, feats_att), dim=0)
    #print(feats_att.shape)
    #print(feats.shape)
    #print(map.shape)
    label = torch.tensor([1,0,0,1])
    #loss = smoothL1loss(map, label)
    #print(loss)


if __name__ == "__main__":
    _test()
