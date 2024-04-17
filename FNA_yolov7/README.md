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

You can download data here. Save the data in the `FNA_yolov7` folder and update `data.yaml` accordingly.
Weights from a trained model can be found here. Save them in the `FNA_yolov7/yolov7` folder.

Download this to train a model from scratch and save it in the `FNA_yolov7/yolov7` folder.
