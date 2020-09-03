cd src
# train
python main.py det_3d --exp_id det_3d --dataset custom_kitti --batch_size 16 --master_batch 7 --num_epochs 2 --lr_step 45,60 --gpus 0 --reg_offset --reg_bbox
# test
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
cd ..
