# ADL Final Project (Shared Task Challenge)

## task description
CinnamonAI對文件定義20類標記，
每一行文字可能無標記、若有標記可能不只一個。

## Create Environment And Dowload Model

`bash download.sh`

## Training

`python3.6 train_extrac_val.py`

- 訓練模型，hyperparameters定義於程式碼內。
- Fine-tuning BERT，另外加上以樂天NLP工具得到的詞性標記。


## Prediction

1. `python3.6 combine_lines.py {config_path} {data_folder}`

    合併同一段落的資料

2. `python3.6 preprocess.py  {config_path} {data_folder}`

將資料處理為模型需要的格式

3. `python3.6 evaluation.py --model_path {model_path} --predict_path {predict_path} --config_path {config_path}`

    輸出預測資料

4. `python3.6 test.py {path_to_test_set_directory}`

    依順序執行前1.2.3.檔案

## requirement

pip install pdf2image # 應該用不到

pip install torch==1.4 torchvision==0.5.0

pip install pytorch-lightning==0.7.6

pip install pandas

pip install transformers

pip install xlrd

pip install rakutenma #https://github.com/rakuten-nlp/rakutenma

bert-base-japanese
https://github.com/cl-tohoku/bert-japanese


## Kaggle Submission Before Competition Closed

### Last submission to Kaggle
Download model
```
bash download2.sh
```
Generate a prediction file `prediction.csv`
```
python3.6 test2.py {path_to_test_set_directory}
```

### TF Bert Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CS1sz6qMJjni2U130kY50lGOYkEWkNWG?usp=sharing)

In the file `tf_bert_model.ipynb`

Use bert model for a simple token classification.

Include:
* Generate more dataset (synonym replacement)
* Preprocessing training data
* TF bert model
* Training process
* Evaluation and Prediction

## dev note
- `train.py`: classification only
- `train_extrac_val.py`: extract label value