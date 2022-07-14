ckpt0='../../../ckpt/text/cv0/ckpt-epoch125.pt'
ckpt1='../../../ckpt/text/cv1/ckpt-epoch120.pt'
ckpt2='../../../ckpt/text/cv2/ckpt-epoch115.pt'
ckpt3='../../../ckpt/text/cv3/ckpt-epoch80.pt'
ckpt4='../../../ckpt/text/cv4/ckpt-epoch80.pt'
data0='../../../data/cross_valid/drugcell_split_1.txt'
data1='../../../data/cross_valid/drugcell_split_2.txt'
data2='../../../data/cross_valid/drugcell_split_3.txt'
data3='../../../data/cross_valid/drugcell_split_4.txt'
data4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/cross_valid_test/text_0/'
output1='hydra.run.dir=./outputs/cross_valid_test/text_1/'
output2='hydra.run.dir=./outputs/cross_valid_test/text_2/'
output3='hydra.run.dir=./outputs/cross_valid_test/text_3/'
output4='hydra.run.dir=./outputs/cross_valid_test/text_4/'
output5='hydra.run.dir=./outputs/cross_valid_test/-1/'

tflag='--cfg job'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
pearson='trainer.epoch_criterion=pearson'

python src/test.py data.test.path=$data0 trainer.ckpt=$ckpt0 $c1 $output0

python src/test.py data.test.path=$data1 trainer.ckpt=$ckpt1 $c1 $output1

python src/test.py data.test.path=$data2 trainer.ckpt=$ckpt2 $c1 $output2
test="""
python src/test.py data.test.path=$data3 trainer.ckpt=$ckpt3 $output3

python src/test.py data.test.path=$data4 trainer.ckpt=$ckpt4 $output4

python src/test.py data.test.path=../../../data/drugcell_all.txt trainer.ckpt=../../../ckpt/dc_v1.pt $output5
"""
