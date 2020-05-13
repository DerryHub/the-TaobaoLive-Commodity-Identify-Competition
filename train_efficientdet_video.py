import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EfficientdetDatasetVideo
from utils import Resizer_video, Normalizer_video, Augmenter_video, collater_video, AdamW
from efficientdet.efficientdet import EfficientDet
import numpy as np
from tqdm import tqdm
from config import get_args_efficientdet

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
                       "collate_fn": collater_video,
                       "num_workers": opt.workers}

    # test_params = {"batch_size": opt.batch_size * num_gpus * 4,
    #                "shuffle": False,
    #                "drop_last": False,
    #                "collate_fn": collater,
    #                "num_workers": opt.workers}

    training_set = EfficientdetDatasetVideo(root_dir=opt.data_path, mode="train",
                               transform=transforms.Compose([Normalizer_video(), Augmenter_video(), Resizer_video()]),
                               maxLen=opt.max_position_embeddings,
                               PAD=opt.PAD)
    training_generator = DataLoader(training_set, **training_params)

    # test_set = EfficientdetDataset(root_dir=opt.data_path, mode="validation",
    #                        transform=transforms.Compose([Normalizer(), Resizer()]))
    # test_generator = DataLoader(test_set, **test_params)

    opt.num_classes = training_set.num_classes
    opt.vocab_size = training_set.vocab_size
    
    model = EfficientDet(opt)
    if opt.resume:
        print('Loading model...')
        model.load_state_dict(torch.load(os.path.join(opt.saved_path, opt.network+'_'+opt.imgORvdo+'.pth')))

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    device = torch.device("cuda:{}".format(device_ids[0]))

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = AdamW(model.parameters(), opt.lr)
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = np.inf
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        model.train()
        epoch_loss = []
        acc = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()
            ins_loss, cls_loss, reg_loss = model([
                data['img'].cuda().float(),
                data['text'].cuda(),
                data['annot'].cuda(), 
            ])
            # cls_loss, reg_loss = model([
            #     data['img'].cuda().float(), 
            #     data['annot'].cuda()
            # ])

            ins_loss = ins_loss.mean()
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss + ins_loss
            if loss == 0:
                continue
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)
            # acc.append(float(acc_))
            # total_acc = np.mean(acc)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            progress_bar.write('Ins loss: {:.5f}\tCls loss: {:.5f}\tReg loss: {:.5f}\tBatch loss: {:.5f}\tTotal loss: {:.5f}'.format(
                    ins_loss, cls_loss, reg_loss, loss, total_loss))
#             progress_bar.write('Ins loss: {:.5f}\tCls loss: {:.5f}\tReg loss: {:.5f}\tBatch loss: {:.5f}\tTotal loss: {:.5f}\n\
# Batch acc: {:.5f}\tTotal acc: {:.5f}'.format(
#                     ins_loss, cls_loss, reg_loss, loss, total_loss, acc_, total_acc))

        scheduler.step(np.mean(epoch_loss))

        # if epoch % opt.test_interval == 0:
        #     model.eval()
        #     loss_regression_ls = []
        #     loss_classification_ls = []
        #     loss_classification_2_ls = []
        #     progress_bar = tqdm(test_generator)
        #     progress_bar.set_description_str(' Evaluating')
        #     for iter, data in enumerate(progress_bar):
        #         with torch.no_grad():
        #             cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])

        #             cls_loss = cls_loss.mean()
        #             reg_loss = reg_loss.mean()

        #             loss_classification_ls.append(float(cls_loss))
        #             loss_regression_ls.append(float(reg_loss))

        #     cls_loss = np.mean(loss_classification_ls)
        #     reg_loss = np.mean(loss_regression_ls)
        #     loss = cls_loss + reg_loss 

        #     print('Epoch: {}/{}. \nClassification loss: {:1.5f}. \tRegression loss: {:1.5f}. \tTotal loss: {:1.5f}'.format(
        #             epoch + 1, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)))

        if total_loss + opt.es_min_delta < best_loss:
            print('Saving model...')
            best_loss = total_loss
            best_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(opt.saved_path, opt.network+'_'+opt.imgORvdo+'.pth'))
            # torch.save(model, os.path.join(opt.saved_path, opt.network+'.pth'))

            # Early stopping
            # if epoch - best_epoch > opt.es_patience > 0:
            #     print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
            #     break


if __name__ == "__main__":
    opt = get_args_efficientdet()
    train(opt)
