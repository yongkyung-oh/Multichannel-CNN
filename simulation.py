#-*- coding:utf-8 -*-

import os
import sys
import time
import datetime
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision
import torchsummary

import pretrainedmodels
import efficientnet_pytorch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms

# Set random seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


# Define dataset
class SensorDataset(Dataset):
    def __init__(self, file_tensor_array, image_transform):
        self.file_tensor_array = file_tensor_array
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.file_tensor_array)
    
    def __getitem__(self, idx):
        label = np.array(self.file_tensor_array[idx, 1]).astype(np.uint8)
        label = torch.tensor(label)
        
        data = torch.load(self.file_tensor_array[idx, 2])
        temp = data[0]
        humi = data[1]
        tensor = data[2]
        tensor = [t.unsqueeze(-1).permute(2,0,1) for t in tensor]
        tensor = [torch.FloatTensor(self.image_transform(transforms.ToPILImage()(t))) for t in tensor]
        tensor = torch.cat(tensor)
        
        sample = {'label': label, 'temp': temp, 'humi': humi, 'tensor': tensor}
        return sample

class TransferModel(nn.Module):
    """
    Transfer model load and setup
    """

    def __init__(self, model_name, transfer, num_input_channel, num_output_class, dropout):
        super(TransferModel, self).__init__()

        # Load model
        # transfer: yes, freeze all, finetune later
        # transfer: no, make trainable all
        if bool(transfer) == True:
            if model_name == 'Xception':
                self.model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
            elif model_name == 'Resnet18':
                self.model = models.resnet18(pretrained=True)
            elif model_name == 'Resnet34':
                self.model = models.resnet34(pretrained=True)
            elif model_name == 'Resnet50':
                self.model = models.resnet50(pretrained=True)
            elif model_name == 'Resnet101':
                self.model = models.resnet101(pretrained=True)
            elif model_name == 'Efficientnet-b0':
                self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
            elif model_name == 'Efficientnet-b4':
                self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')
            elif model_name == 'Efficientnet-b7':
                self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b7')
        else:
            if model_name == 'Xception':
                self.model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained=False)
            elif model_name == 'Resnet18':
                self.model = models.resnet18(pretrained=False)
            elif model_name == 'Resnet34':
                self.model = models.resnet34(pretrained=False)
            elif model_name == 'Resnet50':
                self.model = models.resnet50(pretrained=False)
            elif model_name == 'Resnet101':
                self.model = models.resnet50(pretrained=False)
            elif model_name == 'Efficientnet-b0':
                self.model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b0')
            elif model_name == 'Efficientnet-b4':
                self.model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b4')
            elif model_name == 'Efficientnet-b7':
                self.model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b7')
              
        # Define custom cf / last_linear
        def custom_cf(num_ftrs, num_output_class, dropout):
            #     classifier = nn.Sequential(nn.Linear(int(num_ftrs), int(num_ftrs/2)),
            #                                nn.BatchNorm1d(int(num_ftrs/2)),
            #                                nn.ReLU(), # customize
            #                                nn.Dropout(p=dropout),
            #                                nn.Linear(int(num_ftrs/2), num_out_classes))
            classifier = nn.Sequential(nn.Dropout(p=dropout),
                                       nn.Linear(int(num_ftrs), num_output_class))
            return classifier

        # Change input and last layer
        if model_name in ['Xception']:
            input_layer = nn.Conv2d(num_input_channel, self.model.conv1.out_channels, kernel_size=3, stride=1,
                                    padding=1, dilation=1, groups=1, bias=True)
            self.model.conv1 = input_layer
            self.model.last_linear = custom_cf(self.model.last_linear.in_features, num_output_class, dropout)
        elif model_name in ['Resnet18', 'Resnet34', 'Resnet50', 'Resnet101']:
            input_layer = nn.Conv2d(num_input_channel, self.model.conv1.out_channels, kernel_size=3, stride=1,
                                    padding=1, dilation=1, groups=1, bias=True)
            self.model.conv1 = input_layer
            self.model.fc = custom_cf(self.model.fc.in_features, num_output_class, dropout)
        elif model_name in ['Efficientnet-b0', 'Efficientnet-b4', 'Efficientnet-b7']:
            input_layer = nn.Conv2d(num_input_channel, self.model._conv_stem.out_channels, kernel_size=3, stride=1,
                                    padding=1, dilation=1, groups=1, bias=True)
            self.model._conv_stem = input_layer
            self.model._fc = custom_cf(self.model._fc.in_features, num_output_class, dropout)

        # Set all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, temp, humi, tensor):
        x = self.model(tensor)
        return x

# Set train and evaluate
def train(model, optimizer, criterion, batch_iter):
    model.train()
    corrects, total_loss = 0, 0
    for batch in batch_iter:
        temp = batch['temp'].to(device)
        humi = batch['humi'].to(device)
        tensor = batch['tensor'].to(device)
        y = batch['label'].type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        outputs = model(temp, humi, tensor)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        corrects += (preds == y).sum()
    size = len(batch_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects.item() / size
    return avg_loss, avg_accuracy

def evaluate(model, criterion, batch_iter):
    model.eval()
    corrects, total_loss = 0, 0
    true_list, pred_list = [], []
    with torch.no_grad():
        for batch in batch_iter:
            temp = batch['temp'].to(device)
            humi = batch['humi'].to(device)
            tensor = batch['tensor'].to(device)
            y = batch['label'].type(torch.LongTensor).to(device)

            outputs = model(temp, humi, tensor)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            corrects += (preds == y).sum()
            
            true_list.append(y.tolist())
            pred_list.append(preds.tolist())
    size = len(batch_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects.item() / size
    true_list = [x for y in true_list for x in y]
    pred_list = [x for y in pred_list for x in y]
    return avg_loss, avg_accuracy, true_list, pred_list

def model_run(model_name, transfer, noise):

    print('\nModel: {} | Transfer Learning: {} | Noisy Data: {:.2f}'.format(model_name, transfer, noise))
    
    best_test_loss, best_test_accuracy = None, None

    #path for model
    if not os.path.exists('snapshot'):
        os.mkdir('snapshot')
    if not os.path.exists(os.path.join(os.getcwd(), 'snapshot', '{}_{}'.format(model_name, transfer))):
        os.mkdir(os.path.join(os.getcwd(), 'snapshot', '{}_{}'.format(model_name, transfer)))

    #path for results
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(os.path.join(os.getcwd(), 'results', '{}_{}'.format(model_name, transfer))):
        os.mkdir(os.path.join(os.getcwd(), 'results', '{}_{}'.format(model_name, transfer)))
        
    for i in range(5):
        cv_path = 'tensor_resamples/CV_{}'.format(i)
        print('Cross-Validation with {}'.format(cv_path))
        
        all_file_list = []
        for root, dirs, files in os.walk(os.path.join(cv_path, '{:0.2f}'.format(noise))):
            for file in files:
                all_file_list.append(os.path.join(root, file))

        all_tensor_list = []
        for file_path in all_file_list:
            #noise = file_path.split('/')[2]
            split = file_path.split('/')[3]
            label = file_path.split('/')[4]
            all_tensor_list.append(np.array([split, label, file_path]))
        all_tensor_array = np.array(all_tensor_list)

        train_tensor_array = all_tensor_array[all_tensor_array[:,0]=='train']
        valid_tensor_array = all_tensor_array[all_tensor_array[:,0]=='valid']
        test_tensor_array = all_tensor_array[all_tensor_array[:,0]=='test']
        
        class_weights = torch.FloatTensor(pd.value_counts(train_tensor_array[:,1], normalize = True).sort_index())
        class_weights = torch.FloatTensor([1/w for w in class_weights])
        class_weights = class_weights/class_weights.sum()
        
        image_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                ])

        train_sensor_dataset = SensorDataset(train_tensor_array, image_transform)
        valid_sensor_dataset = SensorDataset(valid_tensor_array, image_transform)
        test_sensor_dataset = SensorDataset(test_tensor_array, image_transform)

        BATCH_SIZE = 4
        train_batch = DataLoader(train_sensor_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_batch = DataLoader(valid_sensor_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        test_batch = DataLoader(test_sensor_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        # Load and setup model
        num_input_channel = 8
        num_output_class = 5
        dropout = 0.5

        model = TransferModel(model_name, transfer, num_input_channel, num_output_class, dropout)
        
        # Train and Evaluate Model
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-8)
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
        # lr scheduler 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        train_out = []
        valid_out = []
        test_out = []
        EPOCHS = 50
        
        best_valid_loss, best_valid_accuracy = None, None
        early_stopping_cnt = 0

        for e in range(0, EPOCHS):
            train_loss, train_accuracy = train(model, optimizer, criterion, train_batch)
            #print("[Epoch: %02d] train loss : %5.5f | train accuracy : %5.5f" % (e+1, train_loss, train_accuracy))

            train_loss, train_accuracy, _ ,_  = evaluate(model, criterion, train_batch)
            #print("[Epoch: %02d] train loss : %5.5f | train accuracy : %03.3f" % (e+1, train_loss, train_accuracy))
            valid_loss, valid_accuracy, _ ,_  = evaluate(model, criterion, valid_batch)
            #print("[Epoch: %02d] valid loss : %5.5f | valid accuracy : %03.3f" % (e+1, valid_loss, valid_accuracy))
            test_loss, test_accuracy, _ ,_  = evaluate(model, criterion, test_batch)
            #print("[Epoch: %02d] test loss  : %5.5f | test accuracy  : %03.3f" % (e+1, test_loss, test_accuracy))
            
            print('[Epoch: {:02d}] train accuracy : {:03.3f} | valid accuracy : {:03.3f} | test accuracy  : {:03.3f}'\
                  .format(e+1, train_accuracy, valid_accuracy, test_accuracy), end="\r", flush=True)
           
            train_out.append([train_loss, train_accuracy])
            valid_out.append([valid_loss, valid_accuracy])
            test_out.append([test_loss, test_accuracy])

            # Early stopping: if loss is not decreasing more than 1% for 5 continuous epochs, then stop it. 
            if e > 0:
                if abs(valid_loss - prev_valid_loss) / prev_valid_loss < 0.01:
                    early_stopping_cnt += 1
                    if early_stopping_cnt == 5:
                        #print("Early stopping at epoch {}".format(e))
                        break
                else:
                    early_stopping_cnt = 0 
            prev_valid_loss = valid_loss
                        
            # Best model selection
            if not best_valid_loss or valid_loss < best_valid_loss:
                #out_name = str('model_weight_{:0.2f}_{}.pt'.format(noise, i))
                out_name = 'model_weight_{:0.2f}.pt'.format(noise)
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'snapshot', '{}_{}'.format(model_name, transfer), out_name))
                best_valid_loss = valid_loss
     
            # Decay Learning Rate
            scheduler.step()
                
        trained_weight = os.path.join(os.getcwd(), 'snapshot', '{}_{}'.format(model_name, transfer), out_name)
        state_dict = torch.load(trained_weight)
        model.load_state_dict(state_dict)
        test_loss, test_accuracy, true_list, pred_list = evaluate(model, criterion, test_batch)
        print("\n test loss : %5.5f | test accuracy : %5.5f" % (test_loss, test_accuracy))
        
        test_result_path = os.path.join(os.getcwd(), 'results', '{}_{}'.format(model_name, transfer), str('test_result_{:0.2f}_{}.pkl'.format(noise, i)))
        with open(test_result_path, 'wb') as f:
            pickle.dump([test_loss, test_accuracy, true_list, pred_list], f)
        
        if not best_test_loss or test_loss < best_test_loss:
            out_name = 'model_weight_{:0.2f}_best.pt'.format(noise)
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'snapshot', '{}_{}'.format(model_name, transfer), out_name))
            best_test_loss = test_loss
            print('\t {} saved'.format(out_name))


if __name__ == '__main__':
    model_name_set = ['Resnet18', 'Resnet34', 'Resnet50']
    noise_set = [0.00, 0.01, 0.03, 0.05]
    
    for transfer in ['True', 'False']:
        for model_name in model_name_set:
            for noise in noise_set:
                model_run(model_name, transfer, noise)