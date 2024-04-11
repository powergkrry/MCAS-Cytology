#!/bin/bash

PW_VALUE=2.0
SOURCE_NAME="obj_pw${PW_VALUE}"
NAME="yolov7-e6e_${SOURCE_NAME}_epoch200"
ROOT="/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_yolov7"
DATA="${ROOT}/data/data.yaml"
HYP="hyp.scratch.p5.${SOURCE_NAME}.yaml"
WEIGHTS="${ROOT}/yolov7/runs/train/${NAME}/weights/best.pt"
IMAGE_SIZE="1536"

if [ "$1" == "train" ]; then
    python3 train_aux.py --device 0 --name $NAME --img $IMAGE_SIZE --data $DATA --weights 'yolov7-e6e_training.pt' --epochs 2 --batch-size 8 --workers 8 --cfg cfg/training/yolov7-e6e.yaml --hyp $HYP --single-cls

elif [ "$1" == "test" ]; then
    python3 test.py --data $DATA --img $IMAGE_SIZE --batch 1 --device 0 --weights $WEIGHTS --name "${NAME}_t" --single-cls --task 'test'

elif [ $# -eq 0 ]; then
    echo "no arguments provided"

else
    echo "invalid argument: $1"

fi