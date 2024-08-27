import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/home/zxw/Desktop/MFAD-RTDETR/runs/train/MFAD-RTDETR/weights/best.pt') # select your model.pt path
    model.predict(source='/home/zxw/Desktop/MFAD-RTDETR/PCB-1386/images/test/01_open_circuit_03.jpg',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )