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

        # predict tag concate(seq,tag)
        self.classifier = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size+1, 20))
        # self.fc_out = nn.Sequential( nn.Linear(self.bert.config.hidden_size + 511, 20))
        # ignore_index 可能用不到
        self.tag_loss = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)

        # for extract answer
        # self.find_start = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 1))
        # self.find_end = nn.Sequential( nn.Dropout(hparams.dropout_rate), nn.Linear(self.bert.config.hidden_size, 1))
        # self.criterion = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)

    def forward(self, input_ids, token_type_ids, attention_mask, pos_tag):

        last_hidden_state, pooler_output, attentions = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # note
        # last_hidden_state: (batch_size, sequence_length, hidden_size)
        # pooler_output: (batch_size, hidden_size)
        # attentions: (batch_size, num_heads, sequence_length, sequence_length)

        # for tag
        pos_tag = pos_tag[:,1:].unsqueeze(2)
        hidden_and_tag = torch.cat((last_hidden_state[:,1:], pos_tag), 2)
        logit = self.classifier(hidden_and_tag)
        return logit

        # for extract answer
        # [batch size, 480, hidden size]
        # logit_start = self.find_start(last_hidden_state[:,1:481])
        # logit_end = self.find_end(last_hidden_state[:,1:481])
   
        # return logit.squeeze(1), logit_start.squeeze(2), logit_end.squeeze(2)

    def _unpack_batch(self, batch):
        return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch["pos_tag"].float(),  batch["tag_n"].long()


    def _calculate_tag_loss(self, y_hat, y):
        # for answerable
        loss = self.tag_loss(y_hat, y)
        return loss


    def _calculate_loss(self, start_hat, end_hat, start, end, token_type_ids, attention_mask):
        # 答案應該在1-480之間 但是cross entropy class不能超過(480-1) 否則下面會噴nan/inf
        src_len = start_hat.size(1)

        # mask out paddings, [cls]/[sep] is already cut
        mask = torch.logical_xor(token_type_ids[:,1:src_len+1], attention_mask[:,1:src_len+1])
        mask = torch.logical_not(mask)

        start_hat = start_hat.masked_fill_(mask, float("-inf"))
        end_hat = end_hat.masked_fill_(mask, float("-inf"))

        loss_start = self.criterion(start_hat, start.long())
        loss_end = self.criterion(end_hat, end.long())

        total_loss = loss_start+loss_end 
        if math.isnan(total_loss) or math.isinf(total_loss):
            print(start)
            print(end)
            print(loss_start)
            print(loss_end)
            # sys.exit()

        return total_loss


    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        logit_tag = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(logit_tag, tag_n)
        return {'loss': loss_tag}

    def validation_step(self, batch, batch_nb):
        input_ids, token_type_ids, attention_mask, pos_tag, tag_n = self._unpack_batch(batch)
        logit_tag = self.forward(input_ids, token_type_ids, attention_mask, pos_tag)

        loss_tag = self._calculate_tag_loss(logit_tag, tag_n)
        # loss = self._calculate_loss(logit_start, logit_end, batch["answer_start"], batch["answer_end"], token_type_ids, attention_mask)
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
    'train_dataset_path': "./dataset/train_2.pkl",
    'valid_dataset_path': "./dataset/dev_2.pkl",
    'batch_size': 4,
    'learning_rate': 0.0005,
    'dropout_rate':0.5,
    'num_workers':4,
    'ignore_index':-1,
    'pos_weight': 0.4,
})

def prediction(args):
    answerable_threshold = args.answerable_threshold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    pretrained_model = BertQA.load_from_checkpoint(checkpoint_path=args.model_path).to(device)
    pretrained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    dataloader = pretrained_model.user_dataloader(args.dataset_path)

    ans = {}
    for batch in tqdm(dataloader):
        answerable, start, end = pretrained_model(batch['input_ids'].to(device), batch['token_type_ids'].to(device), batch['attention_mask'].to(device))
        batch_size = len(batch['id'])
        for index in range(batch_size):
            ans[batch['id'][index]] = extract_ans_text(start[index], end[index], batch['input_ids'][index], tokenizer) if answerable[index].item()>answerable_threshold else ""

    return ans

def token2text(input_ids, start_index, end_index, tokenizer):
    ans_text = ""
    if start_index<=end_index and (end_index-start_index)<=30: # simple case
        ans_token = input_ids[start_index+1:end_index+1]
        ans_text = tokenizer.decode(ans_token).replace(" ","")

    return ans_text

def extract_ans_text(start, end, input_ids, tokenizer):
    # REMOVE PAD skip_special_tokens=TRUE
    ans_text = ""
    p_start, start_index = torch.topk(start, 2)
    p_end, end_index = torch.topk(end, 2)

    answer_probability = []
    answer_probability.append(((p_start[0].item() + p_end[0].item()), start_index[0].item(), end_index[0].item()))
    answer_probability.append(((p_start[1].item() + p_end[0].item()), start_index[1].item(), end_index[0].item()))
    answer_probability.append(((p_start[0].item() + p_end[1].item()), start_index[0].item(), end_index[1].item()))
    answer_probability.append(((p_start[1].item() + p_end[1].item()), start_index[1].item(), end_index[1].item()))
    answer_probability.sort(reverse=True)

    for ans_choice in answer_probability:
        # fisrt token is [CLS]
        start_token, end_token = ans_choice[1]+1, ans_choice[2]+1
        if start_token<=end_token and (end_token-start_token)<=30: # simple case
            ans_token = input_ids[start_token:end_token]
            ans_text = tokenizer.decode(ans_token, skip_special_tokens=True).replace(" ","")
        
        if ans_text!="": break
        
    return ans_text

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bert finetune for qa"
    )

    parser.add_argument('-t','--test_mode', type=bool, required=False,
                        help='test_mode', default=False)

    parser.add_argument('-m','--model_path', type=str, required=False,
                        help='model_path', default="")

    parser.add_argument('-d','--dataset_path', type=str, required=False,
                        help='dataset_path', default="")

    parser.add_argument('-p','--predict_path', type=str, required=False,
                        help='predict_path', default="")

    parser.add_argument('--debug', type=bool, required=False,
                        help='debug info', default=False)

    parser.add_argument('--answerable_threshold', type=float, required=False,
                        help='answerable_threshold', default=0)

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    print(hparams)
    
    if args.test_mode:
        ans_list = prediction(args)

        with open(args.predict_path, 'w') as outfile:
            json_record = json.dumps(ans_list, ensure_ascii=False).encode('utf8').decode()
            outfile.write(json_record)
        
        print('predict done.')

    else:
        print("Training")
        # early_stop_callback = EarlyStopping(verbose=True, mode='max')
        trainer = pl.Trainer(gpus=[0], max_epochs=32, checkpoint_callback=True, early_stop_callback=True)
        bertQA = BertQA(hparams)
        print(bertQA)
        trainer.fit(bertQA)

if __name__ == '__main__':
    args = _parse_args()
    main(args)