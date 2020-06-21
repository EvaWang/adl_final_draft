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

from train import BertQA 
# from train_without_pos import BertQA 

import json
import argparse
from tqdm import tqdm
import csv

# debug
import sys


def prediction(args):

    with open(args.config_path) as f:
        config = json.load(f)

    tag_map = config["tag_map"]
    # answerable_threshold = args.answerable_threshold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    pretrained_model = BertQA.load_from_checkpoint(checkpoint_path=args.model_path).to(device)
    pretrained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)
    dataloader = pretrained_model.user_dataloader(args.dataset_path)

    ans = []
    for batch in tqdm(dataloader):
        hastag, tag = pretrained_model(batch['input_ids'].to(device), batch['token_type_ids'].to(device), batch['attention_mask'].to(device), batch["pos_tag"].float().to(device))
        # tag, val = pretrained_model(batch['input_ids'].to(device), batch['token_type_ids'].to(device), batch['attention_mask'].to(device), batch["pos_tag"].float().to(device))
        batch_size = len(batch['id'])
        for index in range(batch_size):
            tag_name = []
            if hastag[index].item()>0.5:
                tag_name.append("NONE")
            else:
                tags = (tag[index]>0).nonzero().tolist()
                # vals = (val[index]>0.5).nonzero().tolist()
                tag_name = tag_name + [ tag_name for tag_name in tag_map if ([tag_map[tag_name]-1] in tags)]

            ans.append({
                "ID":batch["segment_idx"][index][0][0], 
                "Prediction": " ".join(tag_name),
                "Check": hastag[index].item(),
                "Check2": tag[index].tolist(),
            })

        #     break
        # break

            # top_5_prob, top5_index = torch.topk(tag[index], 5)
            # tag_list = []
            # for i in range(5):
            #     val = top_5_prob[i].item() #TODO
            #     if val <1: break

            #     tag_idx = top5_index[i].item()+1
            #     tag_name = [ t for t in tag_map if tag_idx==tag_map[t]]
            #     tag_list.append(f"{tag_name[-1]}:")
            
            # ans_text = f"{batch['id'][index]},{(' ').join(tag_list)}\n"
            # ans.append(ans_text)
        # break # for test

    return ans



def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bert finetune for qa"
    )

    parser.add_argument('--model_path', type=str, help='model_path', default="./lightning_logs/version_31/checkpoints/epoch=1.ckpt")
    parser.add_argument('--config_path', type=str, help='config_path', default="./dataset/config.json")
    parser.add_argument('--dataset_path', type=str, help='dataset_path', default="./dataset/dev_max100.pkl")
    parser.add_argument('--predict_path', type=str, help='predict_path', default="./test_0621_2.csv")
    parser.add_argument('--debug', type=bool, help='debug info', default=False)
    parser.add_argument('--tag_threshold', type=float, help='tag_threshold', default=0)

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    
    ans_list = prediction(args)

    with open(args.predict_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, ["ID", "Prediction", "Check",  "Check2"])
        writer.writeheader()
        writer.writerows(ans_list)
    
    print('predict done.')

if __name__ == '__main__':
    args = _parse_args()
    main(args)