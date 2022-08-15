import os
import numpy as np
from tqdm import tqdm


def sum_count(file_name):
    return sum(1 for _ in open(file_name))


def main(path, valid_size=5000):
    index_bias = np.array(range(valid_size))
    print(index_bias)
    files = os.listdir(path)
    for ind, f_path in enumerate(files):
        f_path = os.path.join(path, f_path)
        if not os.path.isfile(f_path):
            continue
        train_path = os.path.join(path, 'train', 'train_%s.txt' % f_path[-5])
        val_path = os.path.join(path, 'val', 'val_%s.txt' % f_path[-5])
        train = open(train_path, 'w', encoding='utf-8')
        val = open(val_path, 'w', encoding='utf-8')
        f = open(f_path, 'r', encoding='utf-8')
        f_len = sum_count(f_path)
        index = np.random.randint(0, f_len-valid_size, size=valid_size)
        index = np.sort(index)
        assert len(index) == len(index_bias)
        index = index + index_bias
        for i, line in enumerate(tqdm(f.readlines())):
            if i in index:
                val.write(line)
            else:
                train.write(line)

        train.close()
        val.close()
        f.close()


if __name__ == "__main__":
    path = './data/cross_valid'
    main(path=path)
