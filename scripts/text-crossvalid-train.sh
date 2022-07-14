test0='../../../data/cross_valid/drugcell_split_1.txt'
test1='../../../data/cross_valid/drugcell_split_2.txt'
test2='../../../data/cross_valid/drugcell_split_3.txt'
test3='../../../data/cross_valid/drugcell_split_4.txt'
test4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/text_cross_validation/0/'
output1='hydra.run.dir=./outputs/text_cross_validation/1/'
output2='hydra.run.dir=./outputs/text_cross_validation/2/'
output3='hydra.run.dir=./outputs/text_cross_validation/3/'
output4='hydra.run.dir=./outputs/text_cross_validation/4/'
save0='trainer.save_dir=../../../ckpt/text/cv0'
save1='trainer.save_dir=../../../ckpt/text/cv1'
save2='trainer.save_dir=../../../ckpt/text/cv2'
save3='trainer.save_dir=../../../ckpt/text/cv3'
save4='trainer.save_dir=../../../ckpt/text/cv4'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
lazy='data.train.lazy=true data.val.lazy=true'
notest='data.test=null'
cf='--cfg job'

#nohup python src/train.py +kay=cross ~data.train.path.0 ~data.val.path.0 data.test.path=$test0 $notest $save0 $c0 $output0 > ./outputs/cross_validation0.txt &


#nohup python src/train.py +kay=cross_text ~data.train.path.1 ~data.val.path.1 data.test.path=$test1 $notest $save1 $c0 $output1 > ./outputs/text_cross_validation/1.txt &

#nohup python src/train.py +kay=cross_text ~data.train.path.2 ~data.val.path.2 data.test.path=$test2 $notest $save2 $c1 $output2 > ./outputs/text_cross_validation/2.txt &

nohup python src/train.py +kay=cross_text ~data.train.path.3 ~data.val.path.3 data.test.path=$test3 $notest $save3 $c0 $output3 > ./outputs/text_cross_validation/3.txt &

nohup python src/train.py +kay=cross_text ~data.train.path.4 ~data.val.path.4 data.test.path=$test4 $notest $save4 $c0 $output4 > ./outputs/text_cross_validation/4.txt &
