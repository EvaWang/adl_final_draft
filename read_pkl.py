
import os
import pickle
from BertDataset import BertDataset as dataset
from tqdm import tqdm
from transformers import BertTokenizer

tag_map = {
        "NONE": 0,
        "調達年度": 1,
        "都道府県": 2,
        "入札件名": 3,
        "施設名": 4,
        "需要場所(住所)": 5,
        "調達開始日": 6,
        "調達終了日": 7,	
        "公告日": 8,
        "仕様書交付期限": 9,	
        "質問票締切日時": 10,	
        "資格申請締切日時": 11,	
        "入札書締切日時": 12,
        "開札日時": 13,	
        "質問箇所所属/担当者": 14,
        "質問箇所TEL/FAX": 15,	
        "資格申請送付先": 16,
        "資格申請送付先部署/担当者名": 17,	
        "入札書送付先": 18,	
        "入札書送付先部署/担当者名": 19,
        "開札場所": 20
    }

def read_pkl(set_name):
    data_path = os.path.join(os.getcwd(), f"./dataset/{set_name}.pkl")

    with open(data_path, 'rb') as f:
        bertDataset = pickle.load(f)

    return bertDataset


def count_tag_ratio(dataset):
    for i in range(1, 21):
        poz_val = 0
        count_line = 0
        print(i)
        for data in tqdm(dataset):
            poz_val = poz_val + data["tag_n"][i]
            count_line = count_line+1
        print(f"count_line:{count_line}")
        print(f"poz_val:{poz_val}")
        print(f"weight:{(count_line-poz_val)/poz_val}")
    pass

def check_val(dataset):
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=False)
    print("ID,Prediction,Truth")

    limit = 100
    for data in dataset:
    # for data in tqdm(dataset):
        # print(data["id"])
        # print(data["input_ids"][423:])
        # print(data["token_type_ids"][423:])
        # print(data["attention_mask"][423:])
        # print(data["pos_tag"])
        # print(data["tag"])
        # print(data["tag_n"])
        # print(f'truth:{data["value"]}')
        # print(f'data["start_idx"]:{data["start_idx"]}')
        # print(f'data["end_idx"]:  {data["end_idx"]}')
        # content_ids = data["input_ids"][1:]
        # for s,e in zip(data["start_idx"], data["end_idx"]):
        #     # print(content_ids[s:e+1])
        #     extract_ans = content_ids[s:e+1]
        #     # if len(data["tag"])>0:
        #     #     print(f"s,e:{s},{e}")
        #     #     print(content_ids)
        #     if len(extract_ans)>0 and e!=-1 and s!=-1:
        #         print(f'extracted:{tokenizer.decode(extract_ans, skip_special_tokens=False).replace(" ","")}')
        #     elif e!=-1 and s!=-1:
        #         print(f'{s},{e}, {tokenizer.decode(content_ids, skip_special_tokens=False).replace(" ","")}')
        val_list = data["value"]
        tag_list = data["tag"]
        if len(tag_list) != len(val_list) and len(tag_list)>0:
            val_list = val_list* len(tag_list)

        truth = []
        for tag, val in zip(tag_list, val_list):
            truth.append(f"{tag}:{val}")

        if len(truth) == 0:
            truth.append("NONE")

        predict_list = []
        content_ids = data["input_ids"][1:]
        for i in range(1,21):
            tag_idx = data["tag_n"][i]
            if tag_idx == 1:
                s, e = data["start_idx"][i-1], data["end_idx"][i-1]
                extract_ans = tokenizer.decode(content_ids[s:e+1], skip_special_tokens=True).replace(" ","").replace("##","")
                tag_name = [ tag_name for tag_name in tag_map if (tag_map[tag_name] == i)]
                predict_list.append(f"{tag_name[0]}:{extract_ans}")

        if len(predict_list) == 0:
            predict_list.append("NONE")

        print(f'{data["id"]},{" ".join(predict_list)},{" ".join(truth)}')

        # limit = limit -1
        # if limit <0: break
    pass


if __name__ == '__main__':

    dataset = read_pkl("dev_max_100_2")
    check_val(dataset)
