## 實驗記錄
1. lr:0.001, avg_loss:5.259191036224365,f1: *TODO*, memo: 更新前的dataset, v_num=25
2. lr:0.0005

Python 3.7.5

pip install pdf2image # 應該用不到
pip install torch==1.4 torchvision==0.5.0
pip install pytorch-lightning
pip install pandas
pip install transformers
pip install xlrd
pip install rakutenma #https://github.com/rakuten-nlp/rakutenma

bert-base-japanese
https://github.com/cl-tohoku/bert-japanese

### PoS tag list in Japanese and correspondence to BCCWJ tags

| Tag  | Original JA name | English             |
| ---  | ---------------- | ------------------  |
| A-c  | 形容詞-一般       | Adjective-Common    |
| A-dp | 形容詞-非自立可能  | Adjective-Dependent |
| C    | 接続詞            | Conjunction         |
| D    | 代名詞            | Pronoun             |
| E    | 英単語            | English word        |
| F    | 副詞              | Adverb              |
| I-c  | 感動詞-一般        | Interjection-Common |
| J-c  | 形状詞-一般        | Adjectival Noun-Common |
| J-tari | 形状詞-タリ      | Adjectival Noun-Tari |
| J-xs | 形状詞-助動詞語幹   | Adjectival Noun-AuxVerb stem |
| M-aa | 補助記号-AA        | Auxiliary sign-AA |
| M-c  | 補助記号-一般      | Auxiliary sign-Common |
| M-cp | 補助記号-括弧閉    | Auxiliary sign-Open Parenthesis |
| M-op | 補助記号-括弧開    | Auxiliary sign-Close Parenthesis |
| M-p  | 補助記号-句点      | Auxiliary sign-Period |
| N-n  | 名詞-名詞的        | Noun-Noun |
| N-nc | 名詞-普通名詞      | Noun-Common Noun |
| N-pn | 名詞-固有名詞      | Noun-Proper Noun |
| N-xs | 名詞-助動詞語幹    | Noun-AuxVerb stem |
| O    | その他            | Others            |
| P    | 接頭辞             | Prefix |
| P-fj | 助詞-副助詞        | Particle-Adverbial |
| P-jj | 助詞-準体助詞      | Particle-Phrasal |
| P-k  | 助詞-格助詞        | Particle-Case Marking |
| P-rj | 助詞-係助詞        | Particle-Binding |
| P-sj | 助詞-接続助詞      | Particle-Conjunctive |
| Q-a  | 接尾辞-形容詞的    | Suffix-Adjective |
| Q-j  | 接尾辞-形状詞的    | Suffix-Adjectival Noun |
| Q-n  | 接尾辞-名詞的      | Suffix-Noun |
| Q-v  | 接尾辞-動詞的      | Suffix-Verb |
| R    | 連体詞            | Adnominal adjective |
| S-c  | 記号-一般         | Sign-Common |
| S-l  | 記号-文字         | Sign-Letter  |
| U    | URL              | URL         |
| V-c  | 動詞-一般         | Verb-Common |
| V-dp | 動詞-非自立可能    | Verb-Dependent |
| W    | 空白              | Whitespace |
| X    | 助動詞            | AuxVerb |


['次', 'の', '##と', '##おり', '一', '般', '競', '争', '入', '札', 'に', '付', 'しま', '##す', '。']
['次', 'の', 'とおり',         '一般',     '競争',     '入札',     'に', '付し', 'ます',       '。']
[17,   24,  17,17,17,        17, 17,     17, 17,    17, 17,     24,  35, 35, 38, 38,      15]

[['次', 'N-nc'], ['の', 'P-k'], ['とおり', 'N-nc'], ['一般', 'N-nc'], ['競争', 'N-nc'], ['入札', 'N-nc'], ['に', 'P-k'], ['付し', 'V-c'], ['ます', 'X'], ['。', 'M-p']]