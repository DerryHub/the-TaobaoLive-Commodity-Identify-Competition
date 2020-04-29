import os
import torch
from torch import nn
from torch import optim
from config import get_args_arcface
from dataset import ITMatchTrain
from torch.utils.data import DataLoader
from arcface.head import Arcface
from text.TF_IDF import TF_IDF
from text.BERT import BERT
from text.textcnn import TextCNN
from utils import MSE_match
import numpy as np
from tqdm import tqdm



def train(opt):
    print(opt)
    device_ids = opt.GPUs
    if torch.cuda.is_available():
        num_gpus = len(device_ids)
    else:
        raise Exception('no GPU')

    dataset = ITMatchTrain(opt)
    torch.cuda.empty_cache()
    loader = DataLoader(dataset, num_workers=opt.workers, shuffle=True, drop_last=True, batch_size=opt.batch_size*num_gpus)

    opt.vocab_size = dataset.vocab_size

    if opt.network_text == 'bert':
        model = BERT(opt)
        model_name = 'BERT_'+dataset.model_name
    elif opt.network_text == 'tf_idf':
        model = TF_IDF(opt)
        model_name = 'TFIDF_'+dataset.model_name
    elif opt.network_text == 'textcnn':
        model = TextCNN(opt)
        model_name = 'textcnn_'+dataset.model_name

    head = Arcface(opt)
    head_name = 'arcface_'+dataset.model_name

    if opt.resume:
        print('Loading Models...')
        model.load_state_dict(torch.load(os.path.join(opt.saved_path, model_name+'.pth')))

    head.load_state_dict(torch.load(os.path.join(opt.saved_path, head_name+'.pth')))

    device = torch.device("cuda:{}".format(device_ids[0]))

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    head.to(device)
    head = nn.DataParallel(head, device_ids=device_ids)

    for i in head.parameters():
        i.requires_grad=False

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    optimizer = optim.AdamW([{'params':model.parameters()}, {'params':filter(lambda p: p.requires_grad, head.parameters())}], opt.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    cost = MSE_match()
    cost_arcface = nn.CrossEntropyLoss()

    best_loss = np.inf

    num_iter_per_epoch = len(loader)
    for epoch in range(opt.num_epochs):
        print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
        model.train()
        head.train()
        total = 0
        acc = 0
        epoch_loss = []
        progress_bar = tqdm(loader)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()

            # feature = data['feature'].cuda()
            text = data['text'].cuda()
            instance = data['instance'].cuda()

            text_feature = model(text)
            output = head([text_feature, instance])

            total += instance.size(0)
            acc += (torch.argmax(output, dim=1)==instance).sum().float()

            # loss_mse = cost(text_feature, feature)
            loss_mse = 0
            loss_arcface = cost_arcface(output, instance)
            loss = loss_mse + loss_arcface
            
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)

            progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
            
            progress_bar.write(
                'MSE loss: {:.5f}\tBatch loss: {:.5f}\tTotal loss: {:.5f}\tAccuracy: {:.5f}'.format(
                loss_mse, loss, total_loss, acc/total))

        scheduler.step(np.mean(epoch_loss))

        if total_loss < best_loss:
            print('Saving models...')
            best_loss = total_loss
            torch.save(model.module.state_dict(), os.path.join(opt.saved_path, model_name+'.pth'))

if __name__ == "__main__":
    opt = get_args_arcface()
    train(opt)