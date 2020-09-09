cd src
# train
python main.py det_3d --exp_id det_3d_new_loss --dataset custom_kitti --batch_size 16 --master_batch 16 --num_epochs 20 --lr_step 45,60 --gpus 0 --lr 0.0001 --sc_weight 100
# test
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
cd ..
