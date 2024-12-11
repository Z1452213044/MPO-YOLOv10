from ultralytics import YOLO

model = YOLO("best.pt")#切换你训练出来权重文件的路径
results = model.predict(source=r"C:\Users\zjq\Desktop\train_3.jpg",imgsz=640,conf=0.7,save=True)
save_Dir=results[0].save_dir
print(save_Dir)
