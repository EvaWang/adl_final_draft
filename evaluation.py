import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from BertDataset import BertDataset as dataset
from transformers import BertTokenizer

# from train import BertQA 
from train_extrac_val import BertQA 
# from train_extract_val_without_pos import BertQA 

import json
import argparse
from tqdm import tqdm
import csv

# debug
import sys


def prediction(args):

    with open(args.config_path / 'config.json') as f:
        config = json.load(f)

    tag_map = config["tag_map"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    pretrained_model = BertQA.load_from_checkpoint(checkpoint_path=args.model_path).to(device)
    pretrained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=False)
    dataloader = pretrained_model.user_dataloader(config["preprocessed_filename"])

    ans = []
    # for batch in dataloader:
    for batch in tqdm(dataloader):
        hastag, predict_tag, start, end = pretrained_model(batch['input_ids'].to(device), batch['token_type_ids'].to(device), batch['attention_mask'].to(device), batch["pos_tag"].float().to(device))
        start = start.permute(0,2,1)
        end = end.permute(0,2,1)
        batch_size = len(batch['id'])
        ans_range = []
        for index in range(batch_size):
            tag_and_val = []
            tag_name = []
            ans_item = {}
            
            if hastag[index].item()>0:
                tag_name.append("NONE")
            else:
                tags = (predict_tag[index]>4).nonzero().tolist()
                
                tag_name = tag_name + [ tag_name for tag_name in tag_map if ([tag_map[tag_name]-1] in tags)]
                for t_i, tag in enumerate(tags):
                    p_start, start_index = torch.topk(start[index][tag], 1)
                    p_end, end_index = torch.topk(end[index][tag], 1)
                    content_ids = batch['input_ids'][index][1:]
                    ans_token = content_ids[start_index[0].item():end_index[0].item()+1]
                    ans_text = tokenizer.decode(ans_token, skip_special_tokens=True).replace(" ","").replace("##","")
                    if ans_text:
                        tag_and_val.append(f"{tag_name[t_i]}:{ans_text}")
                        ans_range.append(f"{start_index.item()}-{end_index.item()}")

            if len(tag_and_val) ==0:
                tag_and_val.append("NONE")

            ans_item["ID"] =  batch["id"][index]
            ans_item["Prediction"] =  " ".join(tag_and_val)
                        
            ans.append(ans_item)
    return ans



def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bert finetune for qa"
    )

    parser.add_argument('--model_path', type=str, help='model_path', default="")
    parser.add_argument('--config_path', type=str, help='config_path', default="./dataset/config.json")
    parser.add_argument('--predict_path', type=str, help='predict_path', default="./prediction.csv")

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    
    ans_list = prediction(args)
    csv_header = ["ID", "Prediction"]

    with open(args.predict_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_header)
        writer.writeheader()
        writer.writerows(ans_list)
    
    print('predict done.')

if __name__ == '__main__':
    args = _parse_args()
    main(args)