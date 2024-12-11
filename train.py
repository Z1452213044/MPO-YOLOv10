import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

#下面三个路径根据自己替换！
model_yaml_path=r""#模型配置文件路径
data_yaml_path=r""#数据集路径
pretrained_weights=r"yolov10n.pt"#预训练权重

if __name__ == '__main__':
    model = YOLO(model_yaml_path).load(pretrained_weights)
    model.train(data=data_yaml_path,
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                )

