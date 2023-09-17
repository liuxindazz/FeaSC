python main_pretrain.py \
  -a resnet50 --gamma 0.4 --alpha 0.5\
  -b 512 \
  --dist-url 'tcp://localhost:10501' --multiprocessing-distributed --world-size 1 --rank 0 \
  /dataset/Food2K/ > logs/BYOL_FeaSC.log  2>&1 &