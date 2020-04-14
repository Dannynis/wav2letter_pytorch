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
from datetime import datetime
import torchaudio
import traceback


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,'bartlett':scipy.signal.bartlett}

def load_audio(path):
    # sr, sound = wavfile.read(path)
    sound, sr = librosa.load(path, sr=None)
    # sound = sound.astype('float32') / (2**15 -1)
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    if len(sound) < 10000:
        print ('too short audio {}'.format(path))
        raise Exception
    return sound

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, audio_conf, labels, mel_spec=None, use_cuda=False,setname=None,keep_in_ram=False):
        '''
        Create a dataset for ASR. Audio conf and labels can be re-used from the model.
        Arguments:
            manifest_filepath (string): path to the manifest. Can be a csv containing either path-text pairs, or a pandas DataFrame with columns "filepath" and "text"
            audio_conf (dict): dict containing sample rate, and window size stride and type. 
            labels (list): list containing all valid labels in the text.
            mel_spec(int or None): if not None, use mel spectrogram with that many channels.
            use_cuda(bool): Use torch and torchaudio for stft. Can speed up extraction on GPU.
        '''
        super(SpectrogramDataset, self).__init__()
        prefix_df = pd.read_csv(manifest_filepath,index_col=0,nrows=2)
        if not {'filepath','text'}.issubset(prefix_df.columns) and not {'file_name','text'}.issubset(prefix_df.columns):
            self.df = pd.read_csv(manifest_filepath,header=None,names=['filepath','text'])
        else:
            self.df = pd.read_csv(manifest_filepath,index_col=0)

        if  {'file_name','text'}.issubset(prefix_df.columns):
            self.df['filepath'] = self.df['file_name'].apply(lambda x: os.path.join(os.path.dirname(manifest_filepath), x+'.wav'))
        self.setname = setname
        self.size = len(self.df)
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.use_cuda = use_cuda
        self.mel_spec = mel_spec
        self.n_fft = audio_conf['n_fft']
        self.labels_map = dict([(labels[i],i) for i in range(len(labels))])
        self.validate_sample_rate()
        self.audio_conf = audio_conf
        self.spects = None
        self.keep_in_ram = keep_in_ram
        self.feature_ext_device = 'cuda' if audio_conf['feature_ext_cuda'] else 'cpu'
        #todo export nfft and n mels to configxcc
        self.torch_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                                   n_fft=self.n_fft,hop_length=int(self.n_fft/2),
                                                                   n_mels=self.mel_spec).to(self.feature_ext_device)
        self.preprocess = audio_conf['preprocess_specs']
        if self.audio_conf['preprocess_specs']:
            self.preprocess_spectrograms()

    def async_preprocess_spectrogram(self,x):
        try:
            saved_specs_folder = self.audio_conf['spec_save_folder']
            current_spectrograms_file_name = self.setname + '_'+str(x)
            audio_path = self.df.filepath.iloc[x]
            spect =  self.parse_audio(audio_path)
            with open(os.path.join(saved_specs_folder, current_spectrograms_file_name),'wb') as f:
                pickle.dump(spect, f)
        except:
            print(traceback.format_exc())

        return 0
        #todo add arg optional return to ram

    def load_spect_from_dir(self,index):
        saved_specs_folder = self.audio_conf['spec_save_folder']
        current_spectrograms_file_name = self.setname + '_'+str(index)
        with open(os.path.join(saved_specs_folder, current_spectrograms_file_name),'rb') as f:
            return pickle.load(f)

    def preprocess_spectrograms(self):
            self.spects = {}
            saved_specs_folder = self.audio_conf['spec_save_folder']
            if not os.path.exists(saved_specs_folder):
                os.makedirs(saved_specs_folder)
            saved_specs = os.listdir(saved_specs_folder)
            ## todo add warning if not all the specs have been loaded
            if len(saved_specs) != 0:
                if len(saved_specs) != self.size:
                    print ('missing specs in folder !!!')

                print ('found extracted features')
                if self.keep_in_ram:
                    for i in tqdm.tqdm(range(self.size)):
                        self.spects[i] = self.load_spect_from_dir(i)
            else:
                print ('running multiprocesses spec extraction')
                #todo add args of num preprocess workers
                pool = multiprocessing.dummy.Pool(10)
                async_results = list(tqdm.tqdm(pool.imap(self.async_preprocess_spectrogram,range(self.size)),total=self.size))
                if async_results[0] != 0:
                    self.spects=async_results
                pool.close()
        # for i in tqdm.tqdm(range(self.size)):
        #     audio_path = self.df.filepath.iloc[i]
        #     spect = self.parse_audio(audio_path)
        #     self.spects[i] = spect

    def __getitem__(self, index):
            try:
                sample = self.df.iloc[index]
                audio_path, transcript = sample.filepath, sample.text
                if 'א' in self.labels_map: #Hebrew!
                    import data.language_specific_tools
                    transcript = data.language_specific_tools.hebrew_final_to_normal(transcript)
                #if the specs were generated during preprocess use them otherwise generate during traning
                if self.spects != None and self.spects != {}:
                    spect = self.spects[index]
                elif self.preprocess:
                    spect = self.load_spect_from_dir(index)
                else:
                    spect = self.parse_audio(audio_path)
                target = list(filter(None,[self.labels_map.get(x) for x in list(transcript)]))
                return spect, target, audio_path, transcript
            except Exception as e:
                if index >= len(self.df):
                    raise e
                else:
                    print ('Error on wav {} index {} '.format(audio_path,index))
                    print (traceback.format_exc())
                    print ('Droping..')
                    return None

    def _get_spect(self,audio,n_fft,hop_length,win_length):
        if self.use_cuda: # Use torch based convolutions to compute the STFT
            if self.mel_spec:
                return self.torch_mel_spec(torch.Tensor(audio).to(self.feature_ext_device )).to('cpu')

            e=torch.stft(torch.FloatTensor(audio),n_fft,hop_length,win_length,window=torch.hamming_window(win_length))
            magnitudes = (e ** 2).sum(dim=2) ** 0.5
            return magnitudes
        else: # Use CPU bound libraries
            if self.mel_spec:
                import python_speech_features
                spect, energy = python_speech_features.fbank(audio,samplerate=self.sample_rate,winlen=self.window_size,winstep=self.window_stride,winfunc=np.hamming,nfilt=self.mel_spec)
                return spect.T
            D = librosa.stft(audio, n_fft=n_fft, hop_length = hop_length, win_length=win_length,window=scipy.signal.hamming)
            spect, phase = librosa.magphase(D)

            return spect


    def parse_audio(self,audio_path):
        epsilon = 1e-5
        log_zero_guard_value=2 ** -24
        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        spect = self._get_spect(y,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
        # spect = np.log1p(spect)
        spect = torch.log(spect + log_zero_guard_value)
        #normlize across time
        mean = spect.mean(dim=1)
        std = spect.std(dim=1)
        std += epsilon
        spect = spect - mean.reshape(-1, 1)
        spect = spect / std.reshape(-1, 1)
        return spect
    
    def validate_sample_rate(self):
        audio_path = self.df.iloc[0].filepath
        sr,sound = wavfile.read(audio_path)
        assert sr == self.sample_rate, 'Expected sample rate %d but found %d in first file' % (self.sample_rate,sr)
    
    def __len__(self):
        return self.size

    def data_channels(self):
        '''
        How many channels are returned in each example.
        '''
        return self.mel_spec or int(1+(int(self.sample_rate * self.window_size)/2))

def _collator(batch):
    batch = [sample for sample in batch if sample != None]
    inputs, targets, file_paths, texts = zip(*batch)
    input_lengths = list(map(lambda input: input.shape[1], inputs))
    target_lengths = list(map(len,targets))
    longest_input = max(input_lengths)
    longest_target = max(target_lengths)
    pad_function = lambda x:np.pad(x,((0,0),(0,longest_input-x.shape[1])),mode='constant')
    inputs = torch.FloatTensor(list(map(pad_function,inputs)))
    targets = torch.IntTensor([np.pad(np.array(t),(0,longest_target-len(t)),mode='constant') for t in targets])
    return inputs, input_lengths, targets, target_lengths, file_paths, texts

class BatchAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(BatchAudioDataLoader, self).__init__(*args,**kwargs)
        self.collate_fn = _collator
