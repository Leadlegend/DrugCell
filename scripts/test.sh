test0='../../../data/cross_valid/drugcell_split_1.txt'
output0='./outputs/cross_validation/4/'
lazy='data.train.lazy=true data.val.lazy=true'

python src/train.py +kay=cross ~data.train.path.0 ~data.val.path.0 data.test.path=$test0 \
                    data.train.batch_size=4096 data.test=null hydra.run.dir=$output0
