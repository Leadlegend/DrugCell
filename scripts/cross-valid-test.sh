ckpt0='../../../ckpt/cv0/ckpt-epoch299.pt'
ckpt1='../../../ckpt/cv1/ckpt-epoch263.pt'
ckpt2='../../../ckpt/cv2/ckpt-epoch297.pt'
ckpt3='../../../ckpt/cv3/ckpt-epoch80.pt'
ckpt4='../../../ckpt/cv4/ckpt-epoch80.pt'
data0='../../../data/cross_valid/drugcell_split_1.txt'
data1='../../../data/cross_valid/drugcell_split_2.txt'
data2='../../../data/cross_valid/drugcell_split_3.txt'
data3='../../../data/cross_valid/drugcell_split_4.txt'
data4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/cross_valid_test/0/'
output1='hydra.run.dir=./outputs/cross_valid_test/1/'
output2='hydra.run.dir=./outputs/cross_valid_test/2/'
output3='hydra.run.dir=./outputs/cross_valid_test/3/'
output4='hydra.run.dir=./outputs/cross_valid_test/4/'

tflag='--cfg job'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
pearson='trainer.epoch_criterion=pearson'

python src/test.py data.test.path=$data0 trainer.ckpt=$ckpt0 $output0

python src/test.py data.test.path=$data1 trainer.ckpt=$ckpt1 $output1

python src/test.py data.test.path=$data2 trainer.ckpt=$ckpt2 $output2

python src/test.py data.test.path=$data3 trainer.ckpt=$ckpt3 $output3

python src/test.py data.test.path=$data4 trainer.ckpt=$ckpt4 $output4
