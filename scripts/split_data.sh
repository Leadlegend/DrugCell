data_path='./data/drugcell_all.txt'
mkdir -p ./data/cross_valid/train
mkdir -p ./data/cross_valid/val
cd ./data/cross_valid
split -l 101859 $data_path drugcell_split_
python data/split_trva.py
