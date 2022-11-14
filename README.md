# DCP-NAS: Discrepant Child-Parent Neural Architecture Search for 1-bit CNNs


## Training

sh step1.sh

sh step2.sh

## Evaluating

> python -m torch.distributed.launch --master_port=199999 --nproc_per_node=8 --use_env main.py --genotype=DDPNAS_MCN_2 --load True --epochs 256 --lr 1e-3 --warmup-epochs 0 --init_channels 108 --layers 10 --weight-decay 0 --batch-size 64  --data-path /data/ImageNet/ --output_dir ./DDPNAS_MCN_2_step2_MConv  --resume ./DCP_NAS_Large_best_checkpoint.pth  --eval

Checkpoints and log files can be fetched in https://drive.google.com/drive/folders/1Zs1jSYeF4CYNmByhkfgaEAg0NVJoMoqf?usp=sharing
