data_path='../drugcell_all.txt'
mkdir -p ./data/cross_valid/train
mkdir -p ./data/cross_valid/val
cd ./data/cross_valid
split -l 96865 $data_path drugcell_split_
cd ../..
python data/utils/split_trva.py
