test0='../../../data/cross_valid/drugcell_split_1.txt'
test1='../../../data/cross_valid/drugcell_split_2.txt'
test2='../../../data/cross_valid/drugcell_split_3.txt'
test3='../../../data/cross_valid/drugcell_split_4.txt'
test4='../../../data/cross_valid/drugcell_split_5.txt'
output0='hydra.run.dir=./outputs/text_small/0/'
output1='hydra.run.dir=./outputs/text_small/0_rand/'
output2='hydra.run.dir=./outputs/text_small/0_ver1/'
output3='hydra.run.dir=./outputs/text_cross_validation/3/'
output4='hydra.run.dir=./outputs/text_cross_validation/4/'
save0='trainer.save_dir=../../../ckpt/text/cv0_small'
save1='trainer.save_dir=../../../ckpt/text/cv0_small_rand'
save2='trainer.save_dir=../../../ckpt/text/cv2'
save3='trainer.save_dir=../../../ckpt/text/cv3'
save4='trainer.save_dir=../../../ckpt/text/cv4'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'
cf='--cfg job'
ckpt='trainer.ckpt=../../../ckpt/text/dc_text_rand.pt'
ckpt1='trainer.ckpt=../../../ckpt/text/dc_text_v1.pt'

#python src/train.py +kay=cross_text_small ~data.train.path.0 ~data.val.path.0 $ckpt $save0 $c0 $output0 #--cfg job

#nohup python src/train.py +kay=cross_text_small ~data.train.path.0 ~data.val.path.0 $ckpt $save0 $c0 $output0 > ./outputs/text_small/0.txt &

#nohup python src/train.py +kay=cross_text_small ~data.train.path.0 ~data.val.path.0 $ckpt $save1 $c1 $output1 > ./outputs/text_small/0_rand.txt &

nohup python src/train.py +kay=cross_text_small ~data.train.path.0 ~data.val.path.0 $ckpt1 $save1 $c1 $output2 > ./outputs/text_small/0_ver1.txt &

#nohup python src/train.py +kay=cross_text ~data.train.path.3 ~data.val.path.3 $save3 $c0 $output3 > ./outputs/text_cross_validation/3.txt &

#nohup python src/train.py +kay=cross_text ~data.train.path.4 ~data.val.path.4 $save4 $c0 $output4 > ./outputs/text_cross_validation/4.txt &
