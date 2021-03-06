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
from arcface.resnest_cbam import ResNeStCBAM
from arcface.iresnet import iResNet
from arcface.efficientnet import EfficientNet
from arcface.head import Arcface, LinearLayer, AdaCos, SparseCircleLoss, CombinedMargin
from dataset import ArcfaceDataset, HardTripletDataset
from config import get_args_arcface
from arcface.utils import l2_norm
# from utils import AdamW
from torch.optim import AdamW
import numpy as np
from utils import separate_bn_paras, collater_HardTriplet, CrossEntropyLabelSmooth

def train(opt):
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(opt)
    gpus = list(map(str, opt.GPUs))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
    device_ids = list(range(len(gpus)))
    if torch.cuda.is_available():
        num_gpus = len(device_ids)
    else:
        raise Exception('no GPU')

    cudnn.benchmark = True

    training_params = {"batch_size": opt.batch_size * num_gpus,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": opt.workers}

    training_set = ArcfaceDataset(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    # training_set = HardTripletDataset(
    #     root_dir=opt.data_path, mode="train", size=(opt.size, opt.size), n_samples=opt.n_samples)
    training_generator = DataLoader(training_set, **training_params)

    opt.num_classes = training_set.num_classes
    print(len(training_set))
    print(opt.num_classes)
    # opt.num_labels = training_set.num_labels
    # opt.vocab_size = training_set.vocab_size
    # print(opt.num_classes, opt.vocab_size)
    af = opt.head
    if af == 'arcface':
        head = Arcface(opt)
    elif af == 'adacos':
        head = AdaCos(opt)
    elif af == 'circleloss':
        head = SparseCircleLoss(opt)
    elif af == 'combinedmargin':
        head = CombinedMargin(opt)
    else:
        raise RuntimeError('Cannot Find the Head: {}'.format(opt.head))

    if opt.network == 'resnet':
        backbone = ResNet(opt)
        b_name = opt.network+'_'+opt.mode+'_{}'.format(opt.num_layers_r)
        h_name = af+'_'+b_name
    elif opt.network == 'googlenet':
        backbone = GoogLeNet(opt)
        b_name = opt.network
        h_name = af+'_'+b_name
    elif opt.network == 'inceptionv4':
        backbone = InceptionV4(opt)
        b_name = opt.network
        h_name = af+'_'+b_name
    elif opt.network == 'inceptionresnetv2':
        backbone = InceptionResNetV2(opt)
        b_name = opt.network
        h_name = af+'_'+b_name
    elif opt.network == 'densenet':
        backbone = DenseNet(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_d)
        h_name = af+'_'+b_name
    elif opt.network == 'resnet_cbam':
        backbone = ResNetCBAM(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_c)
        h_name = af+'_'+b_name
    elif opt.network == 'resnest':
        backbone = ResNeSt(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_s)
        h_name = af+'_'+b_name
    elif opt.network == 'iresnet':
        backbone = iResNet(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_i)
        h_name = af+'_'+b_name
    elif opt.network == 'resnest_cbam':
        backbone = ResNeStCBAM(opt)
        b_name = opt.network+'_{}'.format(opt.num_layers_sc)
        h_name = af+'_'+b_name
    elif 'efficientnet' in opt.network:
        backbone = EfficientNet(opt)
        b_name = opt.network
        h_name = af+'_'+b_name
    else:
        raise RuntimeError('Cannot Find the Model: {}'.format(opt.network))

    print('backbone: {}'.format(b_name))
    print('head: {}'.format(h_name))

    if opt.resume:
        print('Loading Backbone Model...')
        backbone.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
        if os.path.isfile(os.path.join(opt.saved_path, h_name+'.pth')):
            print('Loading Head Model...')
            head.load_state_dict(torch.load(os.path.join(opt.saved_path, h_name+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    paras_only_bn, paras_wo_bn = separate_bn_paras(backbone)

    device = torch.device("cuda:{}".format(device_ids[0]))

    backbone.to(device)
    backbone = nn.DataParallel(backbone, device_ids=device_ids)

    head.to(device)
    head = nn.DataParallel(head, device_ids=device_ids)

    # linear.to(device)
    # linear = nn.DataParallel(linear, device_ids=device_ids)

    # optimizer = AdamW([
    #             {'params': backbone.parameters()},
    #             {'params': head.parameters()}
    #         ], opt.lr)
    optimizer = AdamW([
                {'params': paras_wo_bn + [head.module.kernel], 'weight_decay': 5e-4},
                {'params': paras_only_bn}
            ], opt.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    # cost = nn.CrossEntropyLoss()
    cost = CrossEntropyLabelSmooth(opt.num_classes)

    best_loss = np.inf

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        backbone.train()
        head.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        total = 0
        acc = 0
        acc_label = 0
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            img = data['img'].cuda()
            # vdo = data['vdo'].cuda()
            instance = data['instance'].cuda()
            # text = data['text'].cuda()
            # iORv = data['iORv'].cuda()
            # label = data['label'].cuda()

            _, embedding = backbone(img)

            if opt.head == 'circleloss':
                loss, output = head([embedding, instance])
            else:
                output = head([embedding, instance])
                loss = cost(output, instance)

            total += instance.size(0)
            acc += (torch.argmax(output, dim=1)==instance).sum().float()

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
            torch.save(backbone.module.state_dict(), os.path.join(opt.saved_path, b_name+'.pth'))
            torch.save(head.module.state_dict(), os.path.join(opt.saved_path, h_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)