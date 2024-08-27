import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'/home/zxw/Desktop/MFAD-RTDETR/ultralytics/cfg/models/rt-detr/MFAD-RTDETR.yaml')
    model.train(data=r'/home/zxw/Desktop/MFAD-RTDETR/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                workers=8,
                device='0',
                project='runs/train',
                name='MFAD-RTDETR',
                )