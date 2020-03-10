import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from classifier.efficientnet import EfficientNet
from dataset import ClassifierDataset
from tqdm import tqdm
from config import get_args_classifier

opt = get_args_classifier()

if not torch.cuda.is_available():
    raise Exception('no GPU')

cudnn.benchmark = True

train_params = {"batch_size": opt.batch_size,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.workers}

validation_params = {"batch_size": opt.batch_size * 4,
                    "shuffle": False,
                    "drop_last": False,
                    "num_workers": opt.workers}

train_set = ClassifierDataset(size=(opt.image_size, opt.image_size), root_dir=opt.data_path, mode='train')
validation_set = ClassifierDataset(size=(opt.image_size, opt.image_size), root_dir=opt.data_path, mode='validation')

train_generator = DataLoader(train_set, **train_params)
validation_generator = DataLoader(validation_set, **validation_params)

opt.num_classes = train_set.num_classes

model = None
model_name = 'classifier_{}'.format(opt.network)
if opt.resume == True:
    model = EfficientNet.from_name(opt.network, override_params={'num_classes': opt.num_classes})
    print('Loading {}'.format(opt.network))
    model.load_state_dict(torch.load(os.path.join(opt.saved_path, model_name+'.pth')))
else:
    model = EfficientNet.from_pretrained(opt.network, num_classes=opt.num_classes)

if not os.path.isdir(opt.saved_path):
    os.makedirs(opt.saved_path)

model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), opt.lr)

cost = nn.CrossEntropyLoss()
min_loss = np.inf
num_iter_per_epoch = len(train_generator)
for epoch in range(opt.num_epochs):
    print('Epoch: {}/{}:'.format(epoch + 1, opt.num_epochs))
    model.train()
    epoch_loss = []
    progress_bar = tqdm(train_generator)
    total = 0
    acc = 0
    for iter, data in enumerate(progress_bar):
        optimizer.zero_grad()
        imgs = data['img'].cuda()
        labels = data['label'].cuda()

        output = model(imgs)
        total += labels.size(0)
        acc += (torch.argmax(output, dim=1)==labels).sum().float()

        loss = cost(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

        progress_bar.set_description('Epoch: {}/{}. Iteration: {}/{}'.format(
            epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch))
        
        progress_bar.write('Batch loss: {:.5f}\tTotal loss: {:.5f}\tAccuracy: {:.5f}'.format(
            loss, total_loss, acc/total))
    
    model.eval()
    epoch_loss = []
    progress_bar = tqdm(validation_generator)
    total = 0
    acc = 0
    progress_bar.set_description_str('Valiating')
    for iter, data in enumerate(progress_bar):
        imgs = data['img'].cuda()
        labels = data['label'].cuda()

        with torch.no_grad():
            output = model(imgs)

        total += labels.size(0)
        acc += (torch.argmax(output, dim=1)==labels).sum().float()

        loss = cost(output, labels)
        epoch_loss.append(float(loss))
    
    total_loss = np.mean(epoch_loss)

    print('Total loss: {:.5f}\tAccuracy: {:.5f}'.format(total_loss, acc/total))

    if total_loss < min_loss:
        print('Saving model...')
        min_loss = total_loss
        torch.save(model.state_dict(), os.path.join(opt.saved_path, model_name+'.pth'))

    