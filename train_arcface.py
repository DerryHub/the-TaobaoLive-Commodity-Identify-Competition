import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from arcface.backbone import Backbone
from arcface.head import Arcface, LinearLayer
from dataset import ArcfaceDataset
from config import get_args_arcface
import numpy as np

def train(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        raise Exception('no GPU')

    cudnn.benchmark = True

    training_params = {"batch_size": opt.batch_size * num_gpus,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": opt.workers}

    training_set = ArcfaceDataset(root_dir=opt.data_path, mode="train", size=(opt.size, opt.size))
    training_generator = DataLoader(training_set, **training_params)

    opt.num_classes = training_set.num_classes

    backbone = Backbone(opt)
    if opt.pretrain:
        head = LinearLayer(opt)
    else:
        head = Arcface(opt)

    b_name = 'backbone_'+opt.mode+'_{}'.format(opt.num_layers)
    if opt.pretrain:
        h_name = 'linearlayer'
    else:
        h_name = 'arcface_'+opt.mode+'_{}'.format(opt.num_layers)

    print('backbone: {}'.format(b_name))
    print('head: {}'.format(h_name))

    if opt.resume:
        print('Loading model...')
        backbone.load_state_dict(torch.load(os.path.join(opt.saved_path, b_name+'.pth')))
        head.load_state_dict(torch.load(os.path.join(opt.saved_path, h_name+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    backbone = backbone.cuda()
    backbone = nn.DataParallel(backbone)

    head = head.cuda()
    head = nn.DataParallel(head)

    optimizer = torch.optim.AdamW([
                {'params': backbone.parameters()},
                {'params': head.parameters()}
            ], opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    cost = nn.CrossEntropyLoss()

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
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            img = data['img'].cuda()
            label = data['label'].cuda()

            embedding = backbone(img)
            output = head(embedding, label)

            total += label.size(0)
            acc += (torch.argmax(output, dim=1)==label).sum().float()

            loss = cost(output, label)

            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write('Batch loss: {:.5f}\tTotal loss: {:.5f}\tAccuracy: {:.5f}'.format(loss, total_loss, acc/total))

        scheduler.step(np.mean(epoch_loss))

        if total_loss < best_loss:
            print('Saving models...')
            best_loss = total_loss
            torch.save(backbone.module.state_dict(), os.path.join(opt.saved_path, b_name+'.pth'))
            torch.save(head.module.state_dict(), os.path.join(opt.saved_path, h_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)

