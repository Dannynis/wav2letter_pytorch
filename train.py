# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import os.path
import time
import datetime
import random

from data import label_sets
from model import Wav2Letter
from data.data_loader import SpectrogramDataset
from decoder import GreedyDecoder

parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest',help='path to train manifest csv', default='data/train.csv')
parser.add_argument('--val-manifest',help='path to validation manifest csv',default='data/validation.csv')
parser.add_argument('--sample-rate',default=8000,type=int,help='Sample rate')
parser.add_argument('--window-size',default=.02,type=float,help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride',default=.01,type=float,help='Window sstride for spectrogram in seconds')
parser.add_argument('--window',default='hamming',help='Window type for spectrogram generation')
parser.add_argument('--epochs',default=10,type=int,help='Number of training epochs')
parser.add_argument('--lr',default=1e-5,type=float,help='Initial learning rate')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum')
parser.add_argument('--tensorboard',default=True, dest='tensorboard', action='store_true',help='Turn on tensorboard graphing')
parser.add_argument('--no-tensorboard',dest='tensorboard',action='store_false',help='Turn off tensorboard graphing')
parser.add_argument('--log-dir',default='visualize/wav2letter',type=str,help='Directory for tensorboard logs')
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--id',default='Wav2letter training',help='Tensorboard id')
parser.add_argument('--no-model-save',default=False,action='store_true',help='disable model saving entirely')
parser.add_argument('--model-path',default='models/wav2letter.pth.tar')
parser.add_argument('--layers',default=16,type=int,help='Number of Conv1D blocks, between 3 and 16. Last 2 layers are always added.')
parser.add_argument('--labels',default='english',type=str,help='Name of label set to use')
parser.add_argument('--print-samples',default=False,action='store_true',help='Print samples from each epoch')
parser.add_argument('--continue-from',default='',type=str,help='Continue training a saved model')

def get_audio_conf(args):
    audio_conf = {k:args[k] for k in ['sample_rate','window_size','window_stride','window']}
    return audio_conf

def init_new_model(kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    model = Wav2Letter(labels=labels,audio_conf=audio_conf,mid_layers=kwargs['layers'])
    return model

def init_model(kwargs):
    if kwargs['continue_from']:
        model = Wav2Letter.load_model(kwargs['continue_from'])
    else:
        model = init_new_model(kwargs)
    return model

def train(**kwargs):
    print('starting at %s' % time.asctime())
    if kwargs['tensorboard']:
        setup_tensorboard(kwargs['log_dir'])
    model = init_model(kwargs)
    decoder = None#GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(model.audio_conf, kwargs['train_manifest'], model.labels)
    eval_dataset = SpectrogramDataset(model.audio_conf, kwargs['val_manifest'],model.labels)
    criterion = nn.CTCLoss(blank=0,reduction='none')
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters,lr=kwargs['lr'],momentum=kwargs['momentum'],nesterov=True,weight_decay=1e-5)
    model.train()
    epochs=kwargs['epochs']
    print('Train dataset size:%d' % len(train_dataset.ids))
    for epoch in range(epochs):
        total_loss = 0
        index_to_print = random.randrange(len(train_dataset.ids))
        for idx, data in tqdm.tqdm(enumerate(train_dataset)):
            inputs, targets, file_path, text = data
            target_lengths = torch.IntTensor([len(targets)])
            targets = torch.IntTensor(targets).unsqueeze(0)
            out = model(torch.FloatTensor(inputs).unsqueeze(0))
            out = out.transpose(1,0)
            out_sizes = torch.IntTensor([out.size(0)])
            loss = criterion(out,targets,out_sizes,target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            print(loss.data.item())
            if idx == index_to_print and kwargs['print_samples']:
                print('Train case, epoch %d' % epoch)
                print(text)
                print(''.join(map(lambda i: model.labels[i], torch.max(out.squeeze(), 1).indices)))
        log_loss_to_tensorboard(epoch, total_loss / len(train_dataset.ids))
        evaluate(model,eval_dataset,decoder,epoch,kwargs)
    save_model(model, kwargs['model_path'], not kwargs['no_model_save'])
    print('Finished at %s' % time.asctime())

def evaluate(model,dataset,decoder,epoch,kwargs):
    cer, wer = compute_error_rates(model, dataset, decoder, kwargs)
    log_error_rates_to_tensorboard(epoch,cer.mean(),wer.mean())
    
def compute_error_rates(model,dataset,decoder,kwargs):
    with torch.no_grad():
        index_to_print = random.randrange(len(dataset.ids))
        cer = np.zeros(len(dataset.ids))
        wer = np.zeros(len(dataset.ids))
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0))
            out = out.transpose(1,0)
            out_sizes = torch.IntTensor([out.size(0)])
            if idx == index_to_print and kwargs['print_samples']:
                print('Validation case')
                print(text)
                print(''.join(map(lambda i: model.labels[i], torch.max(out.squeeze(), 1).indices)))
            #predicted_texts, offsets = decoder.decode(probs=out.transpose(1,0), sizes=out_sizes)
            #cer[idx] = decoder.cer_ratio(text, predicted_texts[0][0])
            #wer[idx] = decoder.wer_ratio(text, predicted_texts[0][0])
    return cer, wer

_tensorboard_writer = None
def setup_tensorboard(log_dir):
    os.makedirs(log_dir,exist_ok=True)
    from tensorboardX import SummaryWriter
    global _tensorboard_writer
    _tensorboard_writer = SummaryWriter(log_dir)
    
def log_loss_to_tensorboard(epoch,avg_loss):
    print('Total loss: %f' % avg_loss)
    _log_to_tensorboard(epoch,{'Avg Train Loss': avg_loss})
    
def log_error_rates_to_tensorboard(epoch,average_cer,average_wer):
    _log_to_tensorboard(epoch,{'CER': average_cer, 'WER': average_wer})
    
def _log_to_tensorboard(epoch,values,tensorboard_id='Wav2Letter training'):
    if _tensorboard_writer:
        _tensorboard_writer.add_scalars(tensorboard_id,values,epoch+1)
    
def save_model(model,path,should_save):
    if not should_save:
        return
    print('saving model to %s' % path)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    package = Wav2Letter.serialize(model)
    torch.save(package, path)
    
if __name__ == '__main__':
    arguments = parser.parse_args()
    train(**vars(arguments))
