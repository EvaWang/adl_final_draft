import argparse
import logging
import os
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)


def main(args):

    with open(args.config_path / 'config.json') as f:
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
    clean_df_train = normalize_data(df_train)
    del df_train

    logging.info('Creating dataset pickle...')
    content_stack, id2content = process_samples(clean_df_train, config)

    with open("./dataset/train_combine_content.json", "w") as write_file:
        json.dump({"content_list":content_stack,"id2content":id2content}, write_file)


def process_samples(samples, config):

    stack = {}
    id2content = {}
    content = ""
    content_idx = []
    current_page = samples["ID"][0].split('-')[0]
    for i, sample in tqdm(samples.iterrows(), total=samples.shape[0]):
        line_idx = sample["ID"]
        try:
            # 下一個是title則前面內容清空
            is_title = True if type(samples["Is Title"][i+1]) != float else False
        except:
            # 最後一行
            is_title = True

         # 組合同段落 
        line_start = len(content)
        content = content + sample["Text"]
        line_end = len(content)
        content_idx.append([line_idx, line_start, line_end])
        # 組合同段落完畢

        if (is_title or current_page != line_idx.split('-')[0]) and len(content)>0:
            # 先清空前面的content、content_idx
            current_page = line_idx.split('-')[0]
            stack[ f'content_{i}'] = content
            for idx, s, e in content_idx:
                id2content[idx] = {'index': i, 'start':s, 'end':e}
          
            content = ""
            content_idx = []
   
    return stack, id2content


# clean and normalization
def normalize_data(df_train):
    df_train['Text'] = df_train['Text'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)))

    return df_train


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=Path, help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)