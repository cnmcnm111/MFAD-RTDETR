import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'/home/zxw/Desktop/MFAD-RTDETR/runs/train/MFAD-RTDETR/weights/best.pt')
    model.val(data=r'dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=8,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='MFAD-RTDETR',
              )