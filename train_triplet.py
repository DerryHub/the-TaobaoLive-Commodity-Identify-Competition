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
from arcface.densenet import DenseNet
from arcface.resnest import ResNeSt
from arcface.head import Arcface, LinearLayer
from dataset import TripletDataset
from config import get_args_arcface
from arcface.utils import l2_norm
from utils import TripletLoss, TripletAccuracy, TripletFocalLoss, AdamW
import numpy as np
from utils import separate_bn_paras

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
    elif opt.network == 'resnest':
        backbone = ResNeSt(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_s)
        h_name = 'arcface_'+b_name
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    # head = Arcface(opt)

    print('backbone: {}'.format(b_name))
    print('head: {}'.format(h_name))

    if opt.resume:
        print('Loading model...')
        backbone.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
        # if os.path.isfile(os.path.join(opt.saved_path, h_name+'.pth')):
        #     head.load_state_dict(torch.load(os.path.join(opt.saved_path, h_name+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    paras_only_bn, paras_wo_bn = separate_bn_paras(backbone)

    device = torch.device("cuda:{}".format(device_ids[0]))

    backbone.to(device)
    backbone = nn.DataParallel(backbone, device_ids=device_ids)

    # head.to(device)
    # head = nn.DataParallel(head, device_ids=device_ids)

    # optimizer = AdamW([
    #             {'params': paras_wo_bn + [head.module.kernel], 'weight_decay': 5e-4},
    #             {'params': paras_only_bn}
    #         ], opt.lr)
    optimizer = AdamW([
                {'params': paras_wo_bn, 'weight_decay': 5e-4},
                {'params': paras_only_bn}
            ], opt.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    cost_arcface = nn.CrossEntropyLoss()
    cost = TripletLoss(opt.threshold)
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
        total = 0
        acc = 0
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            img_q = data['img_q'].cuda()
            img_p = data['img_p'].cuda()
            img_n = data['img_n'].cuda()
            img_q_instance = data['img_q_instance'].cuda()
            img_p_instance = data['img_p_instance'].cuda()
            img_n_instance = data['img_n_instance'].cuda()

            embedding_q = backbone(img_q)
            embedding_p = backbone(img_p)
            embedding_n = backbone(img_n)

            # output_q = head([embedding_q, img_q_instance])
            # output_p = head([embedding_p, img_p_instance])
            # output_n = head([embedding_n, img_n_instance])

            loss_1 = cost(embedding_q, embedding_p, embedding_n)
            # loss_2_q = cost_arcface(output_q, img_q_instance)
            # loss_2_p = cost_arcface(output_p, img_p_instance)
            # loss_2_n = cost_arcface(output_n, img_n_instance)

            # total += embedding_q.size(0) *3
            # acc += (torch.argmax(output_q, dim=1)==img_q_instance).sum().float()
            # acc += (torch.argmax(output_p, dim=1)==img_p_instance).sum().float()
            # acc += (torch.argmax(output_n, dim=1)==img_n_instance).sum().float()
            acc += 0
            total += 1

            acc_p_, acc_n_, total_p_, total_n_ = accuracy(embedding_q, embedding_p, embedding_n)
            acc_p += acc_p_
            acc_n += acc_n_
            total_p += total_p_
            total_n += total_n_
            
            loss = loss_1# + (loss_2_q + loss_2_p + loss_2_n) / 3
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write(
                'Batch loss: {:.5f}\tTotal loss: {:.5f}\tAccuracy pos: {:.5f}\tAccuracy neg: {:.5f}\tAccuracy: {:.5f}'.format(
                loss, total_loss, acc_p/total_p, acc_n/total_n, acc/total))

        scheduler.step(np.mean(epoch_loss))

        if total_loss < best_loss:
            print('Saving models...')
            best_loss = total_loss
            torch.save(backbone.module.state_dict(), os.path.join(opt.saved_path, b_name+'.pth'))
            # torch.save(head.module.state_dict(), os.path.join(opt.saved_path, h_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)
    
