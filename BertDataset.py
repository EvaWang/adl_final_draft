import torch
from torch.utils.data import Dataset
import random

def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds

class BertDataset(Dataset):
    def __init__(self, data, max_text_len=50):
        # 開頭是[CLS], QA分隔是[SEP], [PAD]補滿
        self.data = data
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = {
            'id': self.data[index]['id'],
            'segment_idx': self.data[index]['segment_idx'],
            'input_ids': self.data[index]["input_ids"],
            'token_type_ids': self.data[index]["token_type_ids"],
            'attention_mask': self.data[index]["attention_mask"],
            'pos_tag': self.data[index]["pos_tag"],
        }

        if 'tag_n' in self.data[index]:
            item['tag'] = self.data[index]['tag']
            item['tag_n'] = self.data[index]['tag_n']
            item['value'] = self.data[index]['value']

        if 'start_idx' in self.data[index]:
            item['start_idx'] = self.data[index]['start_idx']
            item['end_idx'] = self.data[index]['end_idx']

        return item

    def collate_fn(self, samples):
        batch = {}
        key_1 = ['id','segment_idx']
        key2tensor = ['input_ids',"token_type_ids", 'attention_mask', 'pos_tag']
        if 'start_idx' in samples[0]:
            key2tensor.append('start_idx')
            key2tensor.append('end_idx')

        key2pad_tensor = ['pos_tag']

        if 'tag_n' in samples[0]:
            key_1.append('tag')
            # key2tensor.append('value')
            # key2pad_tensor.append('value')

            for sample in samples:
                # has_no_tag = [1]
                # if sum(sample["tag_n"])>0:
                #     has_no_tag = [0]
                # sample["tag_n"] = has_no_tag + sample["tag_n"]
                has_no_tag = 1
                if sum(sample["tag_n"])>0:
                    has_no_tag = 0
                sample["tag_n"][0] = has_no_tag

            batch["tag_n"] = torch.tensor([sample["tag_n"] for sample in samples])
            
        for key in key_1:
            batch[key] = [sample[key] for sample in samples]

        for key in key2tensor:
            batch[key] = [sample[key] for sample in samples]
            if key in key2pad_tensor:
                batch[key] = pad_to_len(batch[key], self.max_text_len) 
            batch[key] = torch.tensor(batch[key])
        
        return batch