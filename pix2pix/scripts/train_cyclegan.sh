set -ex
python train.py --continue_train --dataroot /data/skhare31/edge_birds_cleaned/ --name sobel_rgb_low --model datagan --niter 500 --niter_decay 100
