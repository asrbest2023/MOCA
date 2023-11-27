"""
Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: finetune.py
 - finetune the seed model incorporating the unlabeled instances with pseudo labels
"""

import os
import tarfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import BytesIO
from urllib.request import urlopen

import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss, score_to_weight
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay, ReduceLROnPlateau
from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from mltu.preprocessors import AudioReader
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding

from configs import ModelConfigs

import time
import random
import librosa

configs = ModelConfigs()

# Read data
if configs.dataset == "LJSpeech-1.1":
    pretrained_path = "Models/10_wav2vec2_torch/202310231556/"
elif configs.dataset == "dev-clean":
    pretrained_path = "Models/10_wav2vec2_torch/202311082018/"
unlabeled_dataset = pd.read_csv(pretrained_path + "pred10.csv").values.tolist()
unlabeled_dataset_withlabel = pd.read_csv(pretrained_path + "val.csv").values.tolist()
labeled_dataset = pd.read_csv(pretrained_path + "train.csv").values.tolist()
dataset, test_dataset = [], []

vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

hardnesses = []
ratio = configs.train_ratio

for data in unlabeled_dataset[:int(len(unlabeled_dataset)*ratio)]:
    audio, _ = librosa.load(data[0], sr=8000)
    pred_labels = data[1].split('_')
    scores = data[2].split('_')
    weights = score_to_weight(scores)

    if configs.candidate == True:
        for i in range(configs.top_k):
            idx = random.choices(range(len(pred_labels)), weights=weights)[0]
            selected_element = pred_labels[idx]
            if configs.curriculum == 'speed':
                hardness = len(selected_element)/len(audio) * len(selected_element)
            elif configs.curriculum == 'confidence':
                hardness = len(selected_element) + weights[idx]
            hardnesses.append(hardness)
            dataset.append([data[0], selected_element])
    else:
        pred_labels = data[1].split('_')[0]
        dataset.append([data[0], pred_labels])
        if configs.curriculum == 'speed':
            hardness = len(pred_labels) / len(audio) * len(pred_labels)
        elif configs.curriculum == 'confidence':
            hardness = len(pred_labels) + 1
        hardnesses.append(hardness)

for data in labeled_dataset:
    audio, _ = librosa.load(data[0], sr=8000)
    if configs.curriculum == 'speed':
        hardness = len(data[1])/len(audio) * len(data[1])
    elif configs.curriculum == 'confidence':
        hardness = len(data[1]) + 1
    hardnesses.append(hardness)
    dataset.append(data)

for data in unlabeled_dataset_withlabel[int(len(unlabeled_dataset)*ratio):]:
    test_dataset.append(data)

if configs.sort == True:
    sorted_idx = np.argsort(hardnesses)
    dataset_ordered = [dataset[i] for i in sorted_idx]
else: dataset_ordered = dataset

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=dataset_ordered,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=8000),
        ],
    transformers=[
        LabelIndexer(vocab),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(vocab), use_on_batch=True),
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=configs.train_workers,
    shuffle=False,
)

test_dataProvider = DataProvider(
    dataset=test_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=8000),
        ],
    transformers=[
        LabelIndexer(vocab),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(vocab), use_on_batch=True),
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=configs.train_workers,
)

vocab = sorted(vocab)
configs.vocab = vocab
configs.save()

class CustomWav2Vec2Model(nn.Module):
    def __init__(self, hidden_states, dropout_rate=0.2, **kwargs):
        super(CustomWav2Vec2Model, self).__init__( **kwargs)

        pretrained_name = "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_name, vocab_size=hidden_states, ignore_mismatched_sizes=True)
        self.model.freeze_feature_encoder() # this part does not need to be fine-tuned

    def forward(self, inputs):
        output = self.model(inputs, attention_mask=None).logits
        # Apply softmax
        output = F.log_softmax(output, -1)
        return output

custom_model = CustomWav2Vec2Model(hidden_states = len(vocab)+1)
custom_model.load_state_dict(torch.load(pretrained_path + '/model.pt'))

# put on cuda device if available
if torch.cuda.is_available():
    custom_model = custom_model.to("cuda:0")

# create callbacks
warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    final_lr=configs.final_lr,
    initial_lr=configs.init_lr,
    verbose=True,
)

tb_callback = TensorBoard(configs.model_path + "/logs")
earlyStopping = EarlyStopping(monitor="val_CER", patience=60, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.max_audio_length),
    verbose=1,
    metadata={"vocab": configs.vocab},
    dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}}
)

# create model object that will handle training and testing of the network
model = Model(
    custom_model,
    loss = CTCLoss(blank=len(configs.vocab), zero_infinity=True),
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay),
    metrics=[
        CERMetric(configs.vocab),
        WERMetric(configs.vocab)
    ],
    mixed_precision=configs.mixed_precision,
)

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))

model.fit(
    train_dataProvider,
    test_dataProvider,
    epochs=configs.train_epochs,
    callbacks=[
        warmupCosineDecay,
        tb_callback,
        earlyStopping,
        modelCheckpoint,
        model2onnx
    ]
)