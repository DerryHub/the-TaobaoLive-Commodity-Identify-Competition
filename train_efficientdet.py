import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EfficientdetDataset
from utils import Resizer, Normalizer, Augmenter, collater, AdamW
from efficientdet.efficientdet import EfficientDet
import numpy as np
from tqdm import tqdm
from config import get_args_efficientdet

def train(opt):
    device_ids = opt.GPUs
    if torch.cuda.is_available():
        num_gpus = len(device_ids)
    else:
        raise Exception('no GPU')

    cudnn.benchmark = True

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": opt.workers}

    test_params = {"batch_size": opt.batch_size * num_gpus * 4,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": opt.workers}

    training_set = EfficientdetDataset(root_dir=opt.data_path, mode="train",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    training_generator = DataLoader(training_set, **training_params)

    test_set = EfficientdetDataset(root_dir=opt.data_path, mode="validation",
                           transform=transforms.Compose([Normalizer(), Resizer()]))
    test_generator = DataLoader(test_set, **test_params)

    opt.num_classes = training_set.num_classes

    model = EfficientDet(opt)
    if opt.resume:
        print('Loading model...')
        model.load_state_dict(torch.load(os.path.join(opt.saved_path, opt.network+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    device = torch.device("cuda:{}".format(device_ids[0]))

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = AdamW(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = np.inf
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        model.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()
            cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss
            if loss == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write('Cls loss: {:.5f}\tReg loss: {:.5f}\tBatch loss: {:.5f}\tTotal loss: {:.5f}'.format(
                    cls_loss, reg_loss, loss, total_loss))

        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            loss_classification_2_ls = []
            progress_bar = tqdm(test_generator)
            progress_bar.set_description_str(' Evaluating')
            for iter, data in enumerate(progress_bar):
                with torch.no_grad():
                    cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss 

            print('Epoch: {}/{}. \nClassification loss: {:1.5f}. \tRegression loss: {:1.5f}. \tTotal loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)))

            if loss + opt.es_min_delta < best_loss:
                print('Saving model...')
                best_loss = loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(opt.saved_path, opt.network+'.pth'))
                # torch.save(model, os.path.join(opt.saved_path, opt.network+'.pth'))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break


if __name__ == "__main__":
    opt = get_args_efficientdet()
    train(opt)
