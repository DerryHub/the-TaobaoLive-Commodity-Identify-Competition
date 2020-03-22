import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from arcface.resnet import ResNet
from arcface.googlenet import GoogLeNet
from arcface.inception_v4 import InceptionV4
from arcface.inceptionresnet_v2 import InceptionResNetV2
from dataset import TripletDataset
from config import get_args_arcface
from arcface.utils import l2_norm
from utils import TripletLoss, TripletAccuracy, TripletFocalLoss, AdamW
import numpy as np

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

    training_set = TripletDataset(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    training_generator = DataLoader(training_set, **training_params)

    if opt.network == 'resnet':
        backbone = ResNet(opt)
        b_name = opt.network+'_'+opt.mode+'_{}'.format(opt.num_layers)
    elif opt.network == 'googlenet':
        backbone = GoogLeNet(opt)
        b_name = opt.network
    elif opt.network == 'inceptionv4':
        backbone = InceptionV4(opt)
        b_name = opt.network
    elif opt.network == 'inceptionresnetv2':
        backbone = InceptionResNetV2(opt)
        b_name = opt.network
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    print('backbone: {}'.format(b_name))

    if opt.resume:
        print('Loading model...')
        backbone.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    device = torch.device("cuda:{}".format(device_ids[0]))

    backbone.to(device)
    backbone = nn.DataParallel(backbone)

    optimizer = AdamW([
                {'params': backbone.parameters()}
            ], opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    cost = TripletLoss(opt.alpha, opt.gamma, opt.threshold)
    accuracy = TripletAccuracy()

    best_loss = np.inf

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        backbone.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        total_p = 0
        total_n = 0
        acc_p = 0
        acc_n = 0
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            img_q = data['img_q'].cuda()
            img_p = data['img_p'].cuda()
            img_n = data['img_n'].cuda()

            embedding_q = backbone(img_q)
            embedding_p = backbone(img_p)
            embedding_n = backbone(img_n)

            loss = cost(embedding_q, embedding_p, embedding_n)

            acc_p_, acc_n_, total_p_, total_n_ = accuracy(embedding_q, embedding_p, embedding_n)
            acc_p += acc_p_
            acc_n += acc_n_
            total_p += total_p_
            total_n += total_n_
            
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write(
                'Batch loss: {:.5f}\tTotal loss: {:.5f}\tAccuracy pos: {:.5f}\tAccuracy neg: {:.5f}'.format(
                loss, total_loss, acc_p/total_p, acc_n/total_n))

        scheduler.step(np.mean(epoch_loss))

        if total_loss < best_loss:
            print('Saving models...')
            best_loss = total_loss
            torch.save(backbone.module.state_dict(), os.path.join(opt.saved_path, b_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)
    
