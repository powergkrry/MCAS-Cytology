# Slide-level adenocarcinoma classification from lung specimens
## Implementation of multiple instance learning (MIL) for accurate slide-level automated diagnosis
For train, run
```
python multi_instance_learner.py --epochs 100 --lr_schedule cyclic --experiment_name 9images_mil9_mobilenet_testfold0 --gpu_number 0 --model_name mobilenetv3 --mil_size 9 --test_data_fold (0-3)
```
Before training, make sure to change the fold number to 0, 1, 2, or 3.

You can download data ['data_MIL'](https://figshare.com/ndownloader/files/46557613) and unzip it. Save the data in the `FNA_MIL` folder.
