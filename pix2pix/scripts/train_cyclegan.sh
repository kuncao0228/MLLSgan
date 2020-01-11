set -ex
python train.py --dataroot /data/skhare31/edge_birds_cleaned/ --name sobel_rgb_low --model cycle_gan_notext --niter 500 --niter_decay 100
