# SPDX-FileCopyrightText: Copyright (C) 2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2018-2022 Intel Corporation
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division

import re
import sys

import pyaudio
from six.moves import queue

from argparse import ArgumentParser, SUPPRESS
import logging as log
from time import perf_counter

import pyaudio as pa
import os, time

# Workaround to import librosa on Linux without installed libsndfile.so
try:
    import librosa
except OSError:
    import types
    sys.modules['soundfile'] = types.ModuleType('fake_soundfile')
    import librosa
    
import numpy as np
import scipy
import wave

from openvino.runtime import Core, get_version, PartialShape

#from ctc_decoder import beam_search
#import kenlm
from pyctcdecode import build_ctcdecoder

from struct import pack
import nltk
from nltk import pos_tag
from nltk import word_tokenize
import requests

# Audio recording parameters
#RATE = 16000
#CHUNK = int(RATE / 10)  # 100ms

# duration of signal frame, seconds
FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
CHUNK = CHUNK_SIZE
RATE = SAMPLE_RATE

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

decoder = ''

class QuartzNet:
    pad_to = 16
    alphabet = " abcdefghijklmnopqrstuvwxyz'"

    def __init__(self, core, model_path, input_shape, device):
        assert not input_shape[2] % self.pad_to, f"{self.pad_to} must be a divisor of input_shape's third dimension"
        log.info('Reading model {}'.format(model_path))
        model = core.read_model(model_path)
        if len(model.inputs) != 1:
            raise RuntimeError('QuartzNet must have one input')
        self.input_tensor_name = model.inputs[0].get_any_name()
        model_input_shape = model.inputs[0].shape
        if len(model_input_shape) != 3:
            raise RuntimeError('QuartzNet input must be 3-dimensional')
        if model_input_shape[1] != input_shape[1]:
            raise RuntimeError("QuartzNet input second dimension can't be reshaped")
        if model_input_shape[2] % self.pad_to:
            raise RuntimeError(f'{self.pad_to} must be a divisor of QuartzNet input third dimension')
        if len(model.outputs) != 1:
            raise RuntimeError('QuartzNet must have one output')
        model_output_shape = model.outputs[0].shape
        if len(model_output_shape) != 3:
            raise RuntimeError('QuartzNet output must be 3-dimensional')
        if model_output_shape[2] != len(self.alphabet) + 1:  # +1 for blank char
            raise RuntimeError(f'QuartzNet output third dimension size must be {len(self.alphabet) + 1}')
        
        print("is_dynamic: ", str(model.input(0).partial_shape.is_dynamic))
        model.reshape({self.input_tensor_name: PartialShape(input_shape)})
        print("is_dynamic: ", str(model.input(0).partial_shape.is_dynamic))
        compiled_model = core.compile_model(model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
        log.info('The model {} is loaded to {}'.format(model_path, device))

    def reshape(self, input_shape):
        model.reshape({self.input_tensor_name: PartialShape(input_shape)})
        
    def infer(self, melspectrogram):
        input_data = {self.input_tensor_name: melspectrogram}
        return self.infer_request.infer(input_data)[self.output_tensor]

    @classmethod
    def audio_to_melspectrum(cls, audio, sampling_rate):
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

        win_length = round(sampling_rate * 0.02)
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
            win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))
        mel_basis = librosa.filters.mel(sampling_rate, 512, n_mels=64, fmin=0.0, fmax=8000.0, norm='slaney', htk=False)
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)
        remainder = normalized.shape[1] % cls.pad_to
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, cls.pad_to - remainder)))[None]
        return normalized[None]

# code taken from nvidia nemo sample    
class FrameASR:
    
    def __init__(self, compiled_model, input_tensor_name,
                 frame_len=2, frame_overlap=2, 
                 offset=4):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = [
           " ", 
           "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
           "'",
        ]
        self.vocab.append('_')
        self.input_tensor_name = input_tensor_name
        self.sr = 16000
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = 0.01

        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
                               
        print("self.buffer.shape: {}, per sample: {} ".format( self.buffer.shape , self.buffer.shape[0]/ 16000))
        self.offset = offset
        self.reset()
        
        self.compiled_model = compiled_model
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
        
    def _decode(self, frame, offset=0):

        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame

        melspectrogram = QuartzNet.audio_to_melspectrum(self.buffer, 16000)
        input_data = {self.input_tensor_name: melspectrogram}
        logits = self.infer_request.infer(input_data)[self.output_tensor]

        #decoded = self._greedy_decoder(
        #    logits.squeeze(), 
        #)
        
        decoded = decoder.decode(logits.squeeze(), hotwords=['grill', 'original', 'burger', 'saucy', 'veggie', 'bacon'], hotword_weight=40)

        return decoded[:len(decoded)-offset]
    
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    def _greedy_decoder(self, logits):
        s = ''
        for i in range(logits.shape[0]):
            s += self.vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        sys.stdout.write(response.results + "\r")
        sys.stdout.flush()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.', required=True)
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify the target device to infer on, for example: "
                             "CPU, GPU, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable OpenVINO Runtime plugin for this device. Default value is CPU.")
    return parser
    
    

def main():
    global decoder
    args = build_argparser().parse_args()
    import kenlm
    
    # simple part of speech NLP processor
    nltk.download( 'averaged_perceptron_tagger')
    nltk.download( 'tagsets')
        
    labels = [
          " ", 
          "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
          "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
          "'",
    ]
    
    # it takes time to build the ctc decoder with kenlm - so initialize it early
    # you can find more pretrained kenlm models from https://www.keithv.com/software/giga/ & https://www.keithv.com/software/csr/
    
    decoder = build_ctcdecoder(
          labels,
          'lm_giga_5k_nvp_3gram.arpa', 
          alpha=0.5,  # tuned on a val set 
          beta=1.0,  # tuned on a val set 
    )
    
    model_path = args.model
    device = args.device
    
    core = Core()
    
    log.info('Reading model {}'.format(model_path))
    model = core.read_model(model_path)
    if len(model.inputs) != 1:
                raise RuntimeError('QuartzNet must have one input')
    input_tensor_name = model.inputs[0].get_any_name()
    model_output_shape = model.outputs[0].shape
    model.reshape({input_tensor_name: PartialShape((1,64,512))})   #reshape model to consume longer mfcc frames
    compiled_model = core.compile_model(model, device)
    
    asr = FrameASR(compiled_model, input_tensor_name,
               frame_len=FRAME_LEN, frame_overlap=2, 
               offset=0)

    asr.reset()    
        
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    
    
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()

        requests = ( content for content in audio_generator )
       
        num_of_chars_printed = 0
        for r in requests:
            signal = np.frombuffer(r, dtype=np.int16)
            text = asr.transcribe(signal)

            overwrite_chars = " " * (num_of_chars_printed - len(text))
            num_of_chars_printed = len(text)
            sys.stdout.write(text + overwrite_chars + "\r")
            sys.stdout.flush()

        # Now, put the transcription responses to use.
        #listen_print_loop(responses)

if __name__ == "__main__":
    main()