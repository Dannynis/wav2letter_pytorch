# -*- coding: utf-8 -*-
import librosa
import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import os.path
import time
import math
import datetime
import random
import glob
import multiprocessing
from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder
import timing
from torch.utils.data import BatchSampler, SequentialSampler
from novograd import Novograd
import pickle
                                                                    
parser = argparse.ArgumentParser(description='Wav2Letter training')
parser.add_argument('--train-manifest',help='path to train manifest csv', default='data/train.csv')
parser.add_argument('--val-manifest',help='path to validation manifest csv',default='data/validation.csv')
parser.add_argument('--sample-rate',default=8000,type=int,help='Sample rate')
parser.add_argument('--window-size',default=.02,type=float,help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride',default=.01,type=float,help='Window sstride for spectrogram in seconds')
parser.add_argument('--window',default='hamming',help='Window type for spectrogram generation')
parser.add_argument('--epochs',default=10,type=int,help='Number of training epochs')
parser.add_argument('--lr',default=1e-5,type=float,help='Initial learning rate')
parser.add_argument('--batch-size',default=8,type=int,help='Batch size to use during training')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum')
parser.add_argument('--tensorboard',default=True, dest='tensorboard', action='store_true',help='Turn on tensorboard graphing')
parser.add_argument('--no-tensorboard',dest='tensorboard',action='store_false',help='Turn off tensorboard graphing')
parser.add_argument('--log-dir',default='visualize/wav2letter',type=str,help='Directory for tensorboard logs')
parser.add_argument('--seed',type=int,default=1234)
parser.add_argument('--id',default='Wav2letter training',help='Tensorboard id')
parser.add_argument('--model-dir',default='models/wav2letter',help='Directory to save models. Set as empty, or use --no-model-save to disable saving.')
parser.add_argument('--no-model-save',dest='model_dir',action='store_const',const='')
parser.add_argument('--layers',default=1,type=int,help='Number of Conv1D blocks, between 1 and 16. 2 Additional last layers are always added.')
parser.add_argument('--labels',default='english',type=str,help='Name of label set to use')
parser.add_argument('--print-samples',default=False,action='store_true',help='Print samples from each epoch')
parser.add_argument('--continue-from',default='',type=str,help='Continue training a saved model')
parser.add_argument('--cuda',default=False,action='store_true',help='Enable training and evaluation with GPU')
parser.add_argument('--feature-ext-cuda',default=False,action='store_true',help='Enable feature extraction with GPU')
parser.add_argument('--steps-per-save',default=1000,type=int,help='How many epochs before saving models')
parser.add_argument('--arc',default='quartz',type=str,help='Network architecture to use. Can be either "quartz" (default) or "wav2letter"')
parser.add_argument('--optimizer',default='novograd',type=str,help='Optimizer to use. can be either "sgd" (default) or "novograd". Note that novograd only accepts --lr parameter.')
parser.add_argument('--steps-per-eval',default=200,type=int,help='How many epochs before evaluating the WER and CER')
parser.add_argument('--max-models-history',default=5,type=int,help='How many models to keep saved before overwriting')
parser.add_argument('--preprocess-specs',default=False,action='store_true',help='To process all the spectrograms before trainng')
parser.add_argument('--spec-save-folder',default='./specs_dir',type=str,help='where to dump the specs after preprocessing')
parser.add_argument('--mel-spec-count',default=0,type=int,help='How many channels to use in Mel Spectrogram')
parser.add_argument('--use-mel-spec',dest='mel_spec_count',action='store_const',const=64,help='Use mel spectrogram with default value (64)')
parser.add_argument('--n-fft', default=512, help='size of fft window after padding')
parser.add_argument('--features-in-ram',default=False,action='store_true',help='Store the data features in ram for speedup')

def get_audio_conf(args):
    audio_conf = {k:args[k] for k in ['sample_rate','window_size','window_stride','window'
        ,'preprocess_specs','spec_save_folder','n_fft','feature_ext_cuda']}
    return audio_conf

def init_new_model(arc,channels,kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    model = arc(labels=labels,audio_conf=audio_conf,mid_layers=kwargs['layers'],input_size=channels)
    return model

def init_model(kwargs,channels):
    arcs_map = {"quartz":Jasper,"wav2letter":Wav2Letter}
    arc = arcs_map[kwargs['arc']]
    if kwargs['continue_from']:
        model = arc.load_model(kwargs['continue_from'])
    else:
        model = init_new_model(arc,channels,kwargs)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model contains {} trainable params'.format(pytorch_total_params))
    time.sleep(3)
    print(model)
    return model

def init_datasets(kwargs):
    labels = label_sets.labels_map[kwargs['labels']]
    audio_conf = get_audio_conf(kwargs)
    train_dataset = SpectrogramDataset(kwargs['train_manifest'], audio_conf, labels,mel_spec=kwargs['mel_spec_count']
                                       ,use_cuda=kwargs['cuda'],setname='train',keep_in_ram=kwargs['features_in_ram'])
    batch_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=kwargs['batch_size'], drop_last=False)
    #todo arg of num of worker and warn from cuda and without preprocess
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_sampler=batch_sampler)
    eval_dataset = SpectrogramDataset(kwargs['val_manifest'], audio_conf, labels,
                                      mel_spec=kwargs['mel_spec_count'],use_cuda=kwargs['cuda'],setname='eval',keep_in_ram=kwargs['features_in_ram'])
    eval_batch_loader = BatchAudioDataLoader(eval_dataset, batch_sampler=batch_sampler)


    return train_batch_loader, eval_batch_loader
    
def get_optimizer(params,kwargs):
    if kwargs['optimizer'] == 'sgd':
        return torch.optim.SGD(params,lr=kwargs['lr'],momentum=kwargs['momentum'],nesterov=True,weight_decay=1e-5)
    elif kwargs['optimizer'] == 'novograd':
        return Novograd(params,lr=kwargs['lr'])
    return None

def train(**kwargs):
    print('starting at %s' % time.asctime())
    train_batch_loader, eval_batch_loader = init_datasets(kwargs)
    model = init_model(kwargs,train_batch_loader.dataset.data_channels())
    print('Model and datasets initialized')
    if kwargs['tensorboard']:
        setup_tensorboard(kwargs['log_dir'])
    training_loop(model,kwargs,train_batch_loader, eval_batch_loader)
    if kwargs['tensorboard']:
        _tensorboard_writer.close()

def training_loop(model, kwargs, train_batch_loader, eval_batch_loader):
    device = 'cuda:0' if torch.cuda.is_available() and kwargs['cuda'] else 'cpu'
    model.to(device)
    greedy_decoder = GreedyDecoder(model.labels)
    criterion = nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
    parameters = model.parameters()
    optimizer = get_optimizer(parameters,kwargs)
    scaling_factor = model.get_scaling_factor()
    epochs=kwargs['epochs']
    print('Train dataset size:%d' % len(train_batch_loader.dataset))
    batch_count = math.ceil(len(train_batch_loader.dataset) / kwargs['batch_size'])
    step = 0
    for epoch in range(epochs):
        with timing.EpochTimer(epoch,_log_to_tensorboard) as et:
            model.train()
            total_loss = 0
            for idx, data in et.across_epoch('Data Loading time', tqdm.tqdm(enumerate(train_batch_loader),total=batch_count)):
                inputs, input_lengths, targets, target_lengths, file_paths, texts = data
                with et.timed_action('Model execution time'):
                    model_output = model(torch.FloatTensor(inputs).to(device),input_lengths=torch.IntTensor(input_lengths),)

                    # model_output = model((torch.FloatTensor(inputs).unsqueeze(0).to(device),torch.IntTensor(input_lengths)))

                if type(model) == Jasper:
                    output_lengths = model_output[1]
                    out = model_output[0]
                else:
                    out = model_output
                    output_lengths = torch.Tensor([l // scaling_factor for l in input_lengths]).long()
                out = out.transpose(1,0)
                with et.timed_action('Loss and BP time'):
                    # print(inputs.shape,out.shape,targets.shape,output_lengths,target_lengths)
                    # loss = criterion(out, targets.to(device).long(), torch.IntTensor(output_lengths).long(), torch.IntTensor(target_lengths).long())
                    loss = criterion(out, targets.to(device).long(), output_lengths,
                                     torch.IntTensor(target_lengths).long())
                    if torch.isnan(inputs).any() :
                        print ('BAD BAD INPUTS')
                        continue
                    if torch.isinf(loss).any():
                        print ('loss is inf')
                        continue

                    if torch.isnan(loss).any():
                        print ('losss is nan')
                        continue

                    loss_mean = loss.mean()
                    if idx % 50 == 0:
                        print ('loss {}'.format(loss_mean))
                    optimizer.zero_grad()
                    loss_mean.backward()
                    optimizer.step()

                if step != 0 and step % kwargs['steps_per_save'] == 0 :
                    save_epoch_model(model,step, kwargs['model_dir'])
                if step != 0 and step % int(kwargs['steps_per_eval']) == 0:
                    log_loss_to_tensorboard(epoch, loss_mean)
                    evaluate(model,'train',train_batch_loader, greedy_decoder, epoch, kwargs)
                    evaluate(model,'eval',eval_batch_loader, greedy_decoder, epoch, kwargs)
                step += 1
    if kwargs['model_dir']:
        save_model(model, kwargs['model_dir']+'/final.pth')
    print('Finished at %s' % time.asctime())
    

def evaluate(model,setname,dataset,greedy_decoder,epoch,kwargs):
    greedy_cer, greedy_wer= compute_error_rates(model, dataset, greedy_decoder, kwargs)
    print (setname +' greedy cer:{} greedy wer: {}'.format(greedy_cer.mean(),greedy_wer.mean()))
    log_error_rates_to_tensorboard(epoch,setname,greedy_cer.mean(),greedy_wer.mean())

def compute_error_rates(model,dataset,greedy_decoder,kwargs):
    device = 'cuda:0' if torch.cuda.is_available() and kwargs['cuda'] else 'cpu'
    model.eval()
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        greedy_cer = []
        greedy_wer = []
        original_batch_size = dataset.batch_sampler.batch_size
        #todo export eval batchsize
        dataset.batch_sampler.batch_size = 220
        for idx, (data) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            try:
                inputs, input_lengths, targets, target_lengths, file_paths, texts = data
                model_output = model(torch.FloatTensor(inputs).to(device), input_lengths=torch.IntTensor(input_lengths))
                if type(model) == Jasper:
                    out = model_output[0].to('cpu')
                else:
                    out = model_output.to('cpu')
                greedy_texts = greedy_decoder.decode(probs=out, sizes=target_lengths)
                for text, greedy_text in zip (texts,greedy_texts):
                    greedy_cer.append( greedy_decoder.cer_ratio(text, greedy_text))
                    greedy_wer.append( greedy_decoder.wer_ratio(text, greedy_text))
                if idx< 5:
                    print ('{} original text:{} predicted text:{}'.format(dataset.dataset.setname,text,greedy_text))
            except:
                print ('Error during eval {}'.format(idx))

        dataset.batch_sampler.batch_size = original_batch_size
    return np.array(greedy_cer), np.array(greedy_wer)

_tensorboard_writer = None
def setup_tensorboard(log_dir):
    os.makedirs(log_dir,exist_ok=True)
    from tensorboardX import SummaryWriter
    global _tensorboard_writer
    _tensorboard_writer = SummaryWriter(log_dir)
    
def log_loss_to_tensorboard(epoch,avg_loss):
    print('Batch loss: %f' % avg_loss)
    _log_to_tensorboard(epoch,{'Batch Train Loss': avg_loss})
    
def log_error_rates_to_tensorboard(epoch,set,greedy_cer,greedy_wer):
    _log_to_tensorboard(epoch,{set+'_'+'G_CER': greedy_cer, set+'_'+'G_WER': greedy_wer})
    
def _log_to_tensorboard(epoch,values,tensorboard_id='Wav2Letter training'):
    if _tensorboard_writer:
        _tensorboard_writer.add_scalars(tensorboard_id,values,epoch+1)
    
def save_model(model, path):
    if not path:
        return
    print('saving model to %s' % path)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    package = model.serialize(model)
    torch.save(package, path)
    
def save_epoch_model(model, step, path):
    if not path:
        return
    dirname = os.path.splitext(path)[0]
    model_path = os.path.join(dirname,'step_%d.pth' % step)
    save_model(model, model_path)
    old_files = sorted(glob.glob(dirname+'/step_*'),key=os.path.getmtime,reverse=True)[10:]
    for file in old_files:
        os.remove(file)
    
    
if __name__ == '__main__':
    arguments = parser.parse_args()
    train(**vars(arguments))
