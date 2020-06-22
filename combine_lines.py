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
from rakutenma import RakutenMA


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)

rma = RakutenMA() # (default: phi = 2048, c = 0.003906)


def main(args):

    with open(args.config_path / 'config.json') as f:
        config = json.load(f)

    rma.load(args.config_path / "model_ja.min.json")
    rma.hash_func = rma.create_hash_func(15)
            
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
    content_stack, id2content = combine_lines(clean_df_train, config)

    with open(config["parent_text"], "w") as write_file:
        json.dump({"content_list":content_stack,"id2content":id2content}, write_file)


def combine_lines(samples, config):

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
            tokenized_text_encode = tokenizer.encode_plus(content, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, max_length=config["max_text_len"])
            current_page = line_idx.split('-')[0]
            content_token = tokenizer.tokenize(content)
            tokenized_text_pos = map_pos(content_token, config["pos_map"])

            stack[ f'content_{i}'] = {
                'token':content_token, 
                'tokenized_text_pos':tokenized_text_pos,
                'input_ids': tokenized_text_encode["input_ids"],
                'token_type_ids': tokenized_text_encode["token_type_ids"],
                'attention_mask': tokenized_text_encode["attention_mask"],
                }
            for idx, s, e in content_idx:
                id2content[idx] = {'index': i, 'start':s, 'end':e}
          
            content = ""
            content_idx = []
   
    return stack, id2content


def map_pos(tokenized_text, config_pos_map):
    text_pos = rma.tokenize("".join(tokenized_text))
    pos_indices = []
    current_len = 0
    for pos_token in text_pos:
        current_len = current_len+len(pos_token[0])
        pos_indices.append(current_len)

    # 尋找對應詞性
    tokenized_text_pos = [0] # 第一個token是CLS 擺0
    count_len = 0 # 累積長度
    pos_idx = 0 # 目前pos index
    for token_char in tokenized_text:
        if count_len > pos_indices[pos_idx]:
            pos_idx = pos_idx+1
            
        try:
            pos_tag = config_pos_map[text_pos[pos_idx][1]]
        except:
            pos_tag = 0

        tokenized_text_pos.append(pos_tag)
        if token_char == tokenizer.unk_token:# 遇到[UNK]先加一 看有沒有bug (有)
            count_len = count_len+1
        elif '##' in token_char: # 遇到wordpieces_prefix: [##___]
            count_len = count_len+ len(token_char) -2
        else: 
            count_len = count_len+ len(token_char)

    return tokenized_text_pos
    # 詞性對應完成

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