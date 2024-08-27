import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()