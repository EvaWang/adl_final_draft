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
from collections import deque 

import sys

tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)

def main(args):

    with open(args.output_dir / 'config.json') as f:
        config = json.load(f)
        
    print(f"config:{config}")

    # loading datasets from excel files
    file_list = os.listdir(config.data_folder)
    df_train = []
    df_list = []
    for filepath in tqdm(file_list):
        df = pd.read_excel(os.path.join(config.data_folder, filepath), header=0)
        df_list.append(df)
        
    df_train = pd.concat(df_list, axis=0, ignore_index=True)  
    del df_list


    logging.info('Creating dataset pickle...')
    create_bert_dataset(
        process_samples(df_train,config["max_text_len"], args.is_testset==1),
        args.output_dir / args.output_filename,
        config["max_text_len"]
    )

# clean and normalization
def normalize_data(df_train, config):
    df_train['Tag'] = df_train['Tag'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)).split(';') if pd.notnull(x) else [])
    df_train['Tag_n'] = df_train['Tag'].map(lambda x: [config.tag_map[i] for i in x] if len(x) != 0 else [0])

    df_train['Text'] = df_train['Text'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)))
    df_train['Value'] = df_train['Value'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)) if pd.notnull(x) else '')

    # df_train['y'] = df_train.apply(lambda x: [0]*50, axis=1)

    return df_train

# TODO: read limit from config
def process_samples(samples, max_length=512, is_test=False):
    samples = normalize_data(samples)

    stack = []
    for i, sample in samples.iterrows():
        parent_idx = sample["Parent Index"]
        is_title = sample["Is Title"]
        text = tokenizer.tokenize(sample["Text"])
        tag_n = sample['Tag_n']
        val = []
        if type(df["Value"]) != float: 
            value_list = sample["Value"].split(";")
            val = [tokenizer.tokenize(sub_val) for sub_val in value_list]

        item = (text, tag_n, val)

        if is_title:
            paragraph_text, paragraph_tag

        
        stack.append(item)
            
    return

def find_index():
    return 0



def create_bert_dataset(samples, save_path, config):
    dataset = BertDataset(
        samples
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('is_testset', type=int)
    parser.add_argument('input_filepath', type=Path)
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