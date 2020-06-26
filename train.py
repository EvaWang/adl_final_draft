# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from argparse import Namespace
from typing import Tuple, Dict
import json
import sys
from tqdm import tqdm
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as f

from BertDataset import BertDataset as dataset
from transformers import (
    BertModel,
    BertTokenizer
)

class BertQA(pl.LightningModule):
    def __init__(self, hparams):
        super(BertQA, self).__init__()

        self.hparams = hparams
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese', output_attentions=True)

        # predict tag
        self.classifier = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 21))
        self.tag_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams.pos_weight_tag, dtype=torch.float))


    def forward(self, input_ids, token_type_ids, attention_mask, pos_tag):
        _, pooler_output, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # note
        # last_hidden_state: (batch_size, sequence_length, hidden_size)
        # pooler_output: (batch_size, hidden_size)
        # attentions: (batch_size, num_heads, sequence_length, sequence_length)
   
        # for tag
        tag_logit = self.classifier(pooler_output)
        return tag_logit

    def _unpack_batch(self, batch):
        return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch["pos_tag"].float(),  batch["tag_n"].float()

    def _calculate_tag_loss(self, y_hat, y):
        loss = self.tag_loss(y_hat, y)
        return loss


    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        logit_tag = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(logit_tag, tag_n)
        return {'loss': loss_tag}

    def validation_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        logit_tag = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(logit_tag, tag_n)
        return {'val_loss': loss_tag}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"val avg_loss:{avg_loss} ")

        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # LR scheduler
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) # put LR scheduler here

    def _load_dataset(self, dataset_path: str):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset(self.hparams.train_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                          collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        dataset = self._load_dataset(self.hparams.valid_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn,
                          num_workers=self.hparams.num_workers)

    def user_dataloader(self, dataset_path):
        dataset = self._load_dataset(dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn)

hparams = Namespace(**{
    'train_dataset_path': "./dataset/train_max_512.pkl",
    'valid_dataset_path': "./dataset/dev_max_512.pkl",
    'batch_size': 4,
    'learning_rate': 0.00001,
    'dropout_rate':0.5,
    'num_workers':2,
    'ignore_index':-1,
    'pos_weight_has_tag': [6.92],
    'pos_weight_tag': [6.92, 163.17, 163.17, 116.26, 163.17, 146.01, 132.11, 128.61, 119.12, 184.85, 819.83, 141.75, 222.86, 119.12, 35.08, 79.08, 110.93, 72.51, 75.95, 48.50, 55.61],
    'max_len':100
})

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bert finetune for qa"
    )

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    print(hparams)
    
    trainer = pl.Trainer(gpus=[0], max_epochs=32, gradient_clip_val=2, checkpoint_callback=True, early_stop_callback=True)
#     trainer = pl.Trainer(gpus=[0,1], max_epochs=32, checkpoint_callback=True, early_stop_callback=True)
    bertQA = BertQA(hparams)
    print(bertQA)
    trainer.fit(bertQA)
       

if __name__ == '__main__':
    args = _parse_args()
    main(args)