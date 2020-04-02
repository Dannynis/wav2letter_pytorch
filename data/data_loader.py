# -*- coding: utf-8 -*-

from __future__ import division

import os
import subprocess

import librosa
import numpy as np
import scipy.signal
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import tqdm
import multiprocessing.dummy
import pickle
import datetime
import torchaudio


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,'bartlett':scipy.signal.bartlett}

def load_audio(path):
    sr, sound = wavfile.read(path)
    sound = sound.astype('float32') / (2**15 -1)
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, audio_conf, labels):
        '''
        Create a dataset for ASR. Audio conf and labels can be re-used from the model.
        Arguments:
            manifest_filepath (string): path to the manifest. Can be a csv containing either path-text pairs, or a pandas DataFrame with columns "filepath" and "text"
            audio_conf (dict): dict containing sample rate, and window size stride and type. 
            labels (list): list containing all valid labels in the text.
        '''
        super(SpectrogramDataset, self).__init__()
        prefix_df = pd.read_csv(manifest_filepath,index_col=0,nrows=2)
        if not {'filepath','text'}.issubset(prefix_df.columns) and not {'file_name','text'}.issubset(prefix_df.columns):
            self.df = pd.read_csv(manifest_filepath,header=None,names=['filepath','text'])
        else:
            self.df = pd.read_csv(manifest_filepath,index_col=0)

        if  {'file_name','text'}.issubset(prefix_df.columns):
            self.df['filepath'] = self.df['file_name'].apply(lambda x: os.path.join(os.path.dirname(manifest_filepath), x+'.wav'))

        self.size = len(self.df)
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.labels_map = dict([(labels[i],i) for i in range(len(labels))])
        self.validate_sample_rate()
        self.audio_conf = audio_conf
        self.spects = None
        #todo export nfft and n mels to config
        self.torch_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,n_fft=512,n_mels=64)
        if self.audio_conf['preprocess_specs']:
            self.preprocess_spectrograms()

    def async_preprocess_spectrogram(self,x):
        audio_path = self.df.filepath.iloc[x]
        return self.parse_audio(audio_path)

    def preprocess_spectrograms(self):

        saved_specs_folder = self.audio_conf['spec_save_folder']
        if not os.path.exists(saved_specs_folder):
            os.makedirs(saved_specs_folder)
        saved_specs = os.listdir(saved_specs_folder)
        if len(saved_specs) != 0:
            restored_file_name =saved_specs[-1]
            print('restoring spectrograms from {}'.format(restored_file_name))
            with open(os.path.join(saved_specs_folder,restored_file_name)) as f:
                self.spects = pickle.load(f)
        else:
            print ('running multiprocesses spec extraction')
            pool = multiprocessing.dummy.Pool(4)
            self.spects = list(tqdm.tqdm(pool.imap(self.async_preprocess_spectrogram,range(self.size)),total=self.size))
            current_spectrograms_file_name = str(datetime.now()).replace(' ','_').split('.')[0]
            with open(os.path.join(saved_specs_folder,current_spectrograms_file_name)) as f:
                pickle.dump( self.spects,f)

        # for i in tqdm.tqdm(range(self.size)):
        #     audio_path = self.df.filepath.iloc[i]
        #     spect = self.parse_audio(audio_path)
        #     self.spects[i] = spect

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.filepath, sample.text
        if 'א' in self.labels_map: #Hebrew!
            import data.language_specific_tools
            transcript = data.language_specific_tools.hebrew_final_to_normal(transcript)
        #if the specs were generated during preprocess use them otherwise generate during traning
        if self.spects != None:
            spect = self.spects[index]
        else:
            spect = self.parse_audio(audio_path)
        target = list(filter(None,[self.labels_map.get(x) for x in list(transcript)]))
        return spect, target, audio_path, transcript
    
    def parse_audio(self,audio_path):
        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        
        # D = librosa.stft(y, n_fft=n_fft, hop_length = hop_length, win_length=win_length,window=self.window)
        # spect, phase = librosa.magphase(D)

        # spect = librosa.feature.melspectrogram(y, n_fft=512, hop_length = hop_length,
        #                                    win_length=win_length,window=self.window, n_mels=64)

        spect = self.torch_mel_spec(torch.Tensor(y))
        spect = np.log1p(spect)
        mean = spect.mean()
        std = spect.std()
        spect = np.add(spect,-mean)
        spect = spect / std
        return spect
    
    def validate_sample_rate(self):
        audio_path = self.df.iloc[0].filepath
        sr,sound = wavfile.read(audio_path)
        assert sr == self.sample_rate, 'Expected sample rate %d but found %d in first file' % (self.sample_rate,sr)
    
    def __len__(self):
        return self.size

def _collator(batch):
    inputs, targets, file_paths, texts = zip(*batch)
    input_lengths = list(map(lambda input: input.shape[1], inputs))
    target_lengths = list(map(len,targets))
    longest_input = max(input_lengths)
    longest_target = max(target_lengths)
    pad_function = lambda x:np.pad(x,((0,0),(0,longest_input-x.shape[1])),mode='wrap')
    inputs = torch.FloatTensor(list(map(pad_function,inputs)))
    targets = torch.IntTensor([np.pad(np.array(t),(0,longest_target-len(t)),mode='constant') for t in targets])
    return inputs, input_lengths, targets, target_lengths, file_paths, texts

class BatchAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(BatchAudioDataLoader, self).__init__(*args,**kwargs)
        self.collate_fn = _collator
