import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from transformers import BertTokenizer
from BertDataset import BertDataset
from tqdm import tqdm
import pandas as pd
import unicodedata
import re
from rakutenma import RakutenMA

import sys
# from collections import deque 

rma = RakutenMA() # (default: phi = 2048, c = 0.003906)
rma.load("./dataset/model_ja.min.json")
rma.hash_func = rma.create_hash_func(15)

tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)

def main(args):

    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)
        
    print(f"config:{config}")

    # loading datasets from excel files
    file_list = os.listdir(config["data_folder"])
    df_train = []
    df_list = []
    for filepath in tqdm(file_list):
        filename = os.path.basename(filepath).split('.')[0]
        df = pd.read_excel(os.path.join(config["data_folder"], filepath), header=0)
        df['ID'] = df['Index'].map(lambda x: filename+'-'+str(x))
        df_list.append(df)
        
    df_train = pd.concat(df_list, axis=0, ignore_index=True)  
    del df_list

    # samples = normalize_data(df_train, config)
    # tagged_num = sum([ len(t) for t in df_train['Tag']])
    # total = len(df_train['Tag'])*20
    # print(f"tagged/untagged weight:{(total-tagged_num)/tagged_num}")

    logging.info('Creating dataset pickle...')
    create_bert_dataset(
        process_samples(df_train, config, args.is_testset==1),
        args.output_dir / args.output_filename,
        config["max_text_len"]
    )

# clean and normalization
def normalize_data(df_train, config):
    df_train['Tag'] = df_train['Tag'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)).split(';') if pd.notnull(x) else [])
    df_train['Tag_n'] = df_train['Tag'].map(lambda x: [1 if (type(x) != float and i in x) else 0 for i in config["tag_map"]])

    df_train['Text'] = df_train['Text'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)))
    df_train['Value'] = df_train['Value'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)) if pd.notnull(x) else '')

    df_train['y'] = df_train.apply(lambda x: [0]*50, axis=1)

    return df_train

# TODO: read limit from config
def process_samples(samples, config, is_test=False):
    samples = normalize_data(samples, config)

    stack = []
    for i, sample in tqdm(samples.iterrows(), total=samples.shape[0]):
        # parent_idx = sample["Parent Index"]
        # is_title = sample["Is Title"]
        tokenized_text = tokenizer.tokenize(sample["Text"])
        text_pos = rma.tokenize(sample["Text"])
        val = []
        tag_n = []
        pos = []

        pos_indices = []
        current_len = 0
        for pos_token in text_pos:
            current_len = current_len+len(pos_token[0])
            pos_indices.append(current_len)

        tokenized_text_pos = [0]
        count_len = 0 # 累積長度
        pos_idx = 0 # 目前pos index
        for i, token_char in enumerate(tokenized_text):
            if count_len > pos_indices[pos_idx]:
                pos_idx = pos_idx+1
                
            try:
                pos_tag = config["pos_map"][text_pos[pos_idx][1]]
            except:
                pos_tag = 0

            tokenized_text_pos.append(pos_tag)
            if token_char == tokenizer.unk_token:# 遇到[UNK]先加一 看有沒有bug (有)
                count_len = count_len+1
            elif '##' in token_char: # 遇到wordpieces_prefix: [##___]
                count_len = count_len+ len(token_char) -2
            else: 
                count_len = count_len+ len(token_char)

        if is_test == False:
            tag_n = sample['Tag_n']
            if type(sample["Value"]) != float: 
                value_list = sample["Value"].split(";")
                val = [tokenizer.tokenize(sub_val) for sub_val in value_list]

        tokenized_text_encode = tokenizer.encode_plus(sample["Text"], pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, max_length=config["max_text_len"])
        item = {
            'id': sample["ID"],
            'input_ids': tokenized_text_encode["input_ids"],
            'token_type_ids': tokenized_text_encode["token_type_ids"],
            'attention_mask': tokenized_text_encode["attention_mask"],
            'tag_n': tag_n,
            'tag': sample["Tag"],
            'pos_tag': tokenized_text_pos,
            'value': val # 抽取出來 對應tag的值 先不管
        }
        stack.append(item)

        # TODO: 將段落併在一起訓練
        # if is_title:
            # paragraph_text, paragraph_tag, paragraph_val
            
    return stack


def create_bert_dataset(samples, save_path, config):
    dataset = BertDataset(
        samples
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('is_testset', type=int, default=0,)
    # parser.add_argument('input_filepath', type=Path)
    parser.add_argument('output_dir', type=Path,
                        help='')
    parser.add_argument('output_filename', type=Path,
    help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)