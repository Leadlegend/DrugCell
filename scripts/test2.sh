ckpt0='../../../ckpt/text/cv0/ckpt-epoch50.pt'
ckpt1='../../../ckpt/text/cv0_base/ckpt-epoch50.pt'
ckpt2='../../../ckpt/text/cv0_ver2/ckpt-epoch50.pt'
ckpt3='../../../ckpt/large/cv0_vnn/ckpt-epoch99.pt'
ckpt4='../../../ckpt/text/cv0_ver2.1/ckpt-epoch52.pt'
data0='../../../data/cross_valid/drugcell_split_1.txt'
data1='../../../data/cross_valid/drugcell_split_2.txt'
data2='../../../data/cross_valid/drugcell_split_3.txt'
data3='../../../data/cross_valid/drugcell_split_4.txt'
data4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/cross_valid_test/text0_ver1/'
output1='hydra.run.dir=./outputs/cross_valid_test/text0_base/'
output2='hydra.run.dir=./outputs/cross_valid_test/text0_ver2/'
output3='hydra.run.dir=./outputs/cross_valid_test/large_vnn/'
output4='hydra.run.dir=./outputs/cross_valid_test/text0_ver2.1/'
output5='hydra.run.dir=./outputs/cross_valid_test/-1/'

tflag='--cfg job'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
pearson='trainer.epoch_criterion=pearson'

#python src/test.py data.test.path=$data0 trainer.ckpt=$ckpt0 $output0

#python src/test.py +kay=test data.test.path=$data0 trainer.ckpt=$ckpt0 $output0 $c1

#python src/test.py +kay=test data.test.path=$data0 trainer.ckpt=$ckpt1 $output1 $c1

#python src/test.py +kay=test data.test.path=$data0 trainer.ckpt=$ckpt2 $output2 $c1

#python src/test.py +kay=test_large data.test.path=$data0 trainer.ckpt=$ckpt3 $output3

python src/test.py +kay=test data.test.path=$data0 trainer.ckpt=$ckpt4 $output4
test="""
python src/test.py data.test.path=../../../data/drugcell_all.txt trainer.ckpt=../../../ckpt/dc_v1.pt $output5
"""
