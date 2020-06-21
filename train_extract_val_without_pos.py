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
        self.has_tag = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 1))
        self.classifier = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 20))
        self.has_tag_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams.pos_weight_has_tag, dtype=torch.float), reduction='mean')
        self.tag_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hparams.pos_weight_tag, dtype=torch.float), reduction='none')

        # for extract answer
        self.find_start = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 20))
        self.find_end = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 20))
        self.criterion = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)

    def forward(self, input_ids, token_type_ids, attention_mask, pos_tag):

        last_hidden_state, pooler_output, attentions = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # note
        # last_hidden_state: (batch_size, sequence_length, hidden_size)
        # pooler_output: (batch_size, hidden_size)
        # attentions: (batch_size, num_heads, sequence_length, sequence_length)
   
        # for tag
        has_tag_logit = self.has_tag(pooler_output)
        tag_logit = self.classifier(pooler_output)

        # for val
        # pos_tag = pos_tag[:,1:-1].unsqueeze(2)
        # hidden_and_tag = torch.cat((last_hidden_state[:,1:-1], pos_tag), 2)
        val_start = self.find_start(last_hidden_state[:,1:-1])
        val_end = self.find_end(last_hidden_state[:,1:-1])

        return has_tag_logit.squeeze(1), tag_logit, val_start, val_end

    def _unpack_batch(self, batch):
        return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch["pos_tag"].float(),  batch["tag_n"].float()

    def _calculate_tag_loss(self, y_has_tag_hat, y_hat, y):
        loss_has_tag = self.has_tag_loss(y_has_tag_hat, y[:,0])
        mask = (y[:,0]*(-1)).ge(0).unsqueeze(1).repeat(1,20)
        loss = self.tag_loss(y_hat, y[:,1:])
        loss = torch.masked_select(loss, mask)
        if math.isnan(loss.mean()):
            return loss_has_tag

        return loss_has_tag+loss.mean()

    def _calculate_val_loss(self, start_hat, end_hat, start, end, token_type_ids, attention_mask):

        # mask out paddings, [cls]/[sep] is already cut
        mask = torch.logical_xor(token_type_ids[:,1:-1], attention_mask[:,1:-1])
        mask = torch.logical_not(mask).unsqueeze(2).repeat(1,1,20)

        start_hat = start_hat.masked_fill_(mask, float("-inf"))
        end_hat = end_hat.masked_fill_(mask, float("-inf"))

        loss_start = self.criterion(start_hat, start.long())
        loss_end = self.criterion(end_hat, end.long())

        total_loss = loss_start+loss_end 
        # 答案應該在[0, max_len]之間 cross entropy class 超過噴nan/inf
        if math.isnan(total_loss) or math.isinf(total_loss):
            print(start)
            print(end)
            print(loss_start)
            print(loss_end)
            sys.exit()

        return total_loss

    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        has_tag_logit, logit_tag, val_start, val_end = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)
        # logit_tag, logit_start, logit_end = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(has_tag_logit, logit_tag, tag_n)
        # return {'loss': loss_tag}
        loss_val = self._calculate_val_loss(val_start, val_end, batch["start_idx"], batch["end_idx"], token_type_ids, attention_mask)
        return {'loss': loss_tag+loss_val}

    def validation_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        has_tag_logit, logit_tag, val_start, val_end = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)
        # logit_tag, logit_val = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(has_tag_logit, logit_tag, tag_n)
        # return {'val_loss': loss_tag}
        loss_val = self._calculate_val_loss(val_start, val_end, batch["start_idx"], batch["end_idx"], token_type_ids, attention_mask)
        return {'val_loss': loss_tag+loss_val}

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
    'train_dataset_path': "./dataset/train_max100.pkl",
    'valid_dataset_path': "./dataset/dev_max100.pkl",
    'batch_size': 4,
    'learning_rate': 0.00005,
    'dropout_rate':0.2,
    'num_workers':2,
    'ignore_index':-1,
    'pos_weight_has_tag': [6.92],
    'pos_weight_tag': [163.17, 163.17, 116.26, 163.17, 146.01, 132.11, 128.61, 119.12, 184.85, 819.83, 141.75, 222.86, 119.12, 35.08, 79.08, 110.93, 72.51, 75.95, 48.50, 55.61],
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