# Potential adenocarcinoma detection from lung specimens
## Detection of potential adenocarcinoma in lung specimens using YOLOv7
Dependencies can be installed,
```
pip install -r requirements.txt
```

For train, run
```
./run.sh train
```
For test, run
```
./run.sh test
```

You can download data [`data`](https://figshare.com/ndownloader/files/46557619) and unzip it. Save the data in the `FNA_yolov7` folder and update `data.yaml` accordingly.
Weights from a trained model can be found [`runs`](https://figshare.com/ndownloader/files/46557616). Unzip it. Save them in the `FNA_yolov7/yolov7` folder.

Download [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt) to train a model from scratch and save it in the `FNA_yolov7/yolov7` folder.
