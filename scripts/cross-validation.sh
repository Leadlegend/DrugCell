set -X

test0='../../../data/cross_valid/drugcell_split_1.txt'
test1='../../../data/cross_valid/drugcell_split_2.txt'
test2='../../../data/cross_valid/drugcell_split_3.txt'
test3='../../../data/cross_valid/drugcell_split_4.txt'
test4='../../../data/cross_valid/drugcell_split_5.txt'

nohup python src/train.py +kay=cross ~data.train.path.0 ~data.val.path.0 data.test.path=$test0 --cfg job > ./outputs/cross_validation0.txt &

nohup python src/train.py +kay=cross ~data.train.path.1 ~data.val.path.1 data.test.path=$test1 --cfg job > ./outputs/cross_validation1.txt &

nohup python src/train.py +kay=cross ~data.train.path.2 ~data.val.path.2 data.test.path=$test2 --cfg job > ./outputs/cross_validation2.txt &

nohup python src/train.py +kay=cross ~data.train.path.3 ~data.val.path.3 data.test.path=$test3 --cfg job > ./outputs/cross_validation3.txt &

nohup python src/train.py +kay=cross ~data.train.path.4 ~data.val.path.4 data.test.path=$test4 --cfg job > ./outputs/cross_validation4.txt &
