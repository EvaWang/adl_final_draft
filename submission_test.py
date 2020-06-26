import combine_lines 
import preprocess 
import test
import argparse
from pathlib import Path

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bert finetune for qa"
    )

    parser.add_argument('data_folder', type=Path, help='./release2/test/ca_data')
    parser.add_argument('--config_path', type=Path, help='config_path', default="./dataset")
    parser.add_argument('--model_path', type=str, help='model_path', default="./lightning_logs/version_63/checkpoints/epoch=1.ckpt")
    parser.add_argument('--predict_path', type=str, help='predict_path', default="./prediction.csv")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    combine_lines.main(args)
    preprocess.main(args)
    test.main(args)