import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from arcface.resnet import ResNet
from arcface.googlenet import GoogLeNet
from arcface.inception_v4 import InceptionV4
from arcface.inceptionresnet_v2 import InceptionResNetV2
from arcface.densenet import DenseNet
from arcface.resnet_cbam import ResNetCBAM
from arcface.resnest import ResNeSt
from arcface.efficientnet import EfficientNet
from arcface.head import Arcface, LinearLayer
from dataset import ArcfaceDatasetSeparate
from config import get_args_arcface
from arcface.utils import l2_norm
# from utils import AdamW
from torch.optim import AdamW
import numpy as np
from utils import separate_bn_paras, collater_HardTriplet, MSE_match

def train(opt):
    print(opt)
    device_ids = opt.GPUs
    if torch.cuda.is_available():
        num_gpus = len(device_ids)
    else:
        raise Exception('no GPU')

    cudnn.benchmark = True

    training_params = {"batch_size": opt.batch_size * num_gpus,
                        "shuffle": True,
                        "drop_last": True,
                        "num_workers": opt.workers}

    training_set = ArcfaceDatasetSeparate(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    training_generator = DataLoader(training_set, **training_params)

    opt.num_classes = training_set.num_classes

    if opt.network == 'resnet':
        backbone = ResNet(opt)
        b_name = opt.network+'_'+opt.mode+'_{}'.format(opt.num_layers_r)
        h_name = 'arcface_'+b_name
    elif opt.network == 'googlenet':
        backbone = GoogLeNet(opt)
        b_name = opt.network
        h_name = 'arcface_'+b_name
    elif opt.network == 'inceptionv4':
        backbone = InceptionV4(opt)
        b_name = opt.network
        h_name = 'arcface_'+b_name
    elif opt.network == 'inceptionresnetv2':
        backbone = InceptionResNetV2(opt)
        b_name = opt.network
        h_name = 'arcface_'+b_name
    elif opt.network == 'densenet':
        backbone = DenseNet(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_d)
        h_name = 'arcface_'+b_name
    elif opt.network == 'resnet_cbam':
        backbone_image = ResNetCBAM(opt)
        backbone_video = ResNetCBAM(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_c)
        h_name = 'arcface_'+b_name
        b_name_image = b_name + '_image'
        b_name_video = b_name + '_video'
    elif opt.network == 'resnest':
        backbone_image = ResNeSt(opt)
        backbone_video = ResNeSt(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_s)
        h_name = 'arcface_'+b_name
        b_name_image = b_name + '_image'
        b_name_video = b_name + '_video'
    elif 'efficientnet' in opt.network:
        backbone = EfficientNet(opt)
        b_name = opt.network
        h_name = 'arcface_'+b_name
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    head = Arcface(opt)

    print('backbone_image: {}'.format(b_name_image))
    print('backbone_video: {}'.format(b_name_video))
    print('head: {}'.format(h_name))

    if opt.resume:
        print('Loading Backbone Model...')
        backbone_image.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name_image+'.pth')))
        backbone_video.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name_video+'.pth')))
        if os.path.isfile(os.path.join(opt.saved_path, h_name+'.pth')):
            print('Loading Head Model...')
            head.load_state_dict(torch.load(os.path.join(opt.saved_path, h_name+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    paras_only_bn_image, paras_wo_bn_image = separate_bn_paras(backbone_image)
    paras_only_bn_video, paras_wo_bn_video = separate_bn_paras(backbone_video)

    device = torch.device("cuda:{}".format(device_ids[0]))

    backbone_image.to(device)
    backbone_image = nn.DataParallel(backbone_image, device_ids=device_ids)

    backbone_video.to(device)
    backbone_video = nn.DataParallel(backbone_video, device_ids=device_ids)

    head.to(device)
    head = nn.DataParallel(head, device_ids=device_ids)

    optimizer = AdamW([
                {'params': paras_wo_bn_image + [head.module.kernel] + paras_wo_bn_video, 'weight_decay': 5e-4},
                {'params': paras_only_bn_image + paras_only_bn_video}
            ], opt.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    cost = nn.CrossEntropyLoss()
    cost_mse = MSE_match()

    best_loss = np.inf

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        backbone_image.train()
        backbone_video.train()
        head.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        total = 0
        acc = 0
        acc_label = 0
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            img = data['img'].cuda()
            vdo = data['vdo'].cuda()
            instance = data['instance'].cuda()

            embedding_image = backbone_image(img)
            embedding_video = backbone_video(vdo)

            output_image = head([embedding_image, instance])
            output_video = head([embedding_video, instance])

            total += instance.size(0)*2
            acc += (torch.argmax(output_image, dim=1)==instance).sum().float()
            acc += (torch.argmax(output_video, dim=1)==instance).sum().float()
            # acc_label += (torch.argmax(label_output, dim=1)==label).sum().float()

            loss_mse = cost_mse(embedding_image, embedding_video)
            loss = (cost(output_image, instance) + cost(output_video, instance))/2 + loss_mse

            loss_head = 0
            loss_all = loss
            loss_all.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write(
                'Batch loss: {:.5f}\tTotal loss: {:.5f}\tHead loss: {:.5f}\tAccuracy: {:.5f}'.format(
                loss, total_loss, loss_head, acc/total))

        scheduler.step(np.mean(epoch_loss))

        if total_loss < best_loss:
            print('Saving models...')
            best_loss = total_loss
            torch.save(backbone_image.module.state_dict(), os.path.join(opt.saved_path, b_name_image+'.pth'))
            torch.save(backbone_video.module.state_dict(), os.path.join(opt.saved_path, b_name_video+'.pth'))
            torch.save(head.module.state_dict(), os.path.join(opt.saved_path, h_name+'.pth'))
            # torch.save(linear.module.state_dict(), os.path.join(opt.saved_path, l_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)



