# EMDetect

Deep learning to detect particular features on EM images.  
1. Defect detection (fold/cracks)
2. Resin detection
3. Film detection


### Train
```
python train.py --exp_dir /experiment_directory/ 
--train_image /train_image.h5 --train_label /train_mask.h5
--val_image /val_image.h5 --val_label /val_mask.h5 --chkpt_num 0
```

### Inference
```
python inference.py --exp_dir /experiment_directory/ --chkpt_num 150000 
--input_file test_image.h5 --output_file pred_mask.h5
```

### Large-scale inference
Large-scale inference can be done using the [SEAMLeSS](https://github.com/seung-lab/SEAMLeSS) module. \
Use *alex-fold_detector* branch.
```
python run_fold_detector.py --model_path ../models/alex_FoldNet0106_124k/ 
--src_path gs://seunglab_minnie_phase3/alignment/unaligned --dst_path gs://neuroglancer/alex/fold_detection/minnie/FoldNet0106_124k_unaligned 
--bbox_start 0 0 17000 --bbox_stop 491520 491520 17080 --bbox_mip 0 --max_mip 8 --mip 4 --chunk_size 2048 2048 --overlap 32 32
```
