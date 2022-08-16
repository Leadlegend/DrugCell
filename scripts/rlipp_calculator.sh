ckpt0='trainer.ckpt=../../../ckpt/baseline/cv0/ckpt-epoch50.pt'

output0='hydra.run.dir=./outputs/rlipp/base_pre/'
output1='hydra.run.dir=./outputs/rlipp/base/'

save0='trainer.save_dir=../../../data/rlipp/feature'
pred_final='pred=../../../data/rlipp/feature/Final.txt'

tc='--cfg job'
c0='trainer.device=cuda:0'
c1='trainer.device=cuda:1'

#python src/rlipp_pre.py $ckpt0 $output0 $c0

nohup python src/rlipp.py $output1 > ./p.log &
