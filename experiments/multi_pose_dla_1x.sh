cd src
# train
python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16

python main.py multi_pose --exp_id multipose_dla_1x --dataset coco_hp --batch_size 8 --master_batch 8 --lr 5e-4  --gpus 0 --num_workers 4 --load_model ../models/multi_pose_dla_1x.pth

python demo.py multi_pose --exp_id multipose_dla_1x --dataset coco_hp --demo ../images/img.jpg --load_model ../models/trained_models/multi_pose_dla_1x.pth

# test
python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume
# flip test
python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume --flip_test
cd ..
