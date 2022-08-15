test0='../../../data/cross_valid/drugcell_split_1.txt'
test1='../../../data/cross_valid/drugcell_split_2.txt'
test2='../../../data/cross_valid/drugcell_split_3.txt'
test3='../../../data/cross_valid/drugcell_split_4.txt'
test4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/base_cross_validation/0_new/'
output1='hydra.run.dir=./outputs/base_cross_validation/1_new/'
output2='hydra.run.dir=./outputs/base_cross_validation/2/'
output3='hydra.run.dir=./outputs/base_cross_validation/3/'
output4='hydra.run.dir=./outputs/base_cross_validation/4/'
save0='trainer.save_dir=../../../ckpt/baseline/cv0_new'
save1='trainer.save_dir=../../../ckpt/baseline/cv1_new'
save2='trainer.save_dir=../../../ckpt/baseline/cv2'
save3='trainer.save_dir=../../../ckpt/baseline/cv3'
save4='trainer.save_dir=../../../ckpt/baseline/cv4'
ckpt0='trainer.ckpt=../../../ckpt/baseline/cv0/ckpt-epoch35.pt'
ckpt1='trainer.ckpt=../../../ckpt/baseline/cv1/ckpt-epoch35.pt'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
notest='data.test=null'

#python src/train.py +kay=cross ~data.train.path.0 ~data.val.path.0 $save0 $c1 $output0

#nohup python src/train.py +kay=cross ~data.train.path.0 ~data.val.path.0 $save0 $c1 $output0 > ./outputs/base_cross_validation/0_new.txt &

nohup python src/train.py +kay=cross ~data.train.path.1 ~data.val.path.1 $save1 $c0 $output1 > ./outputs/base_cross_validation/1_new.txt &

#nohup python src/train.py +kay=cross ~data.train.path.2 ~data.val.path.2 $notest $save2 $c0 $output2 > ./outputs/base_cross_validation/2.txt &

#nohup python src/train.py +kay=cross ~data.train.path.3 ~data.val.path.3 $notest $save3 $c0 $output3 > ./outputs/base_cross_validation/3.txt &

#nohup python src/train.py +kay=cross ~data.train.path.4 ~data.val.path.4 $notest $save4 $c0 $output4 > ./outputs/base_cross_validation/4.txt &
