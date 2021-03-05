import os
import sys
home_dir = os.getcwd()
if not home_dir in sys.path:
    sys.path.append(home_dir)

from tqdm import tqdm
import pandas as pd


def get_clean_str(strs):
    strs = [item for item in strs if u'\u4e00' <= item <= u'\u9fa5' or item in ['!', '?']]
    return ''.join(strs)


def clean_lcqmc(file, save_file):
    df = pd.read_csv(file, header=None, sep='\t')
    p_list = df.iloc[:, 0].values
    h_list = df.iloc[:, 1].values
    label_list = df.iloc[:, 2].values

    new_df = []
    for p, h, label in tqdm(zip(p_list, h_list, label_list)):
        new_p = get_clean_str(p)
        new_h = get_clean_str(h)
        if len(new_p) > 3 and len(new_h) > 3 and int(label) in [0, 1]:
            new_df.append((new_p, new_h, label))
    new_df = pd.DataFrame(data=new_df)
    new_df.to_csv(save_file, index=False, header=False, sep='\t')


if __name__ == '__main__':
    train_lcqmc_file = os.path.join(home_dir, 'data/lcqmc/train.txt')
    test_lcqmc_file = os.path.join(home_dir, 'data/lcqmc/test.txt')
    dev_lcqmc_file = os.path.join(home_dir, 'data/lcqmc/dev.txt')
    clean_train_file = os.path.join(home_dir, 'data/clean_lcqmc/train.txt')
    clean_test_file = os.path.join(home_dir, 'data/clean_lcqmc/test.txt')
    clean_dev_file = os.path.join(home_dir, 'data/clean_lcqmc/dev.txt')

    clean_lcqmc(train_lcqmc_file, clean_train_file)
    clean_lcqmc(test_lcqmc_file, clean_test_file)
    clean_lcqmc(dev_lcqmc_file, clean_dev_file)
    print('-----清洗完毕-----')