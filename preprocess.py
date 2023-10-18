import cv2
import os
import base64
import torch
from modules.detect import Detector
from modules.crop import crop_image
import numpy as np
from PIL import Image

def get_image(path:str):
    
    filelist = os.listdir(path)
    print(filelist)
    print(cv2.__version__)
    print(" ======= Parsing Video data is : ", path)

    cnt = 0

    for clip in filelist:
        video = cv2.VideoCapture(f'video/{clip}')

        if not video.isOpened():
            print("Could not Open :", clip)
            exit(0)
        
        print(" ======= Parsing Video data is : ", clip)
        
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
    
        print("length :", length)
        print("width :", width)
        print("height :", height)
        print("fps :", fps, "\n")
        
        while(video.isOpened()):
            ret, img = video.read()
            if(int(video.get(1)% int(fps) == 0)):
                cv2.imwrite(f'image/{cnt}.jpg', img)
                print("imagefile number : ", f'{cnt}.jpg')
                cnt += 1
            if(ret ==  False):
                break
        video.release()


def human_detection(path):
    human_detector = Detector()
    
    with torch.no_grad():
        for filename in os.listdir('image'):
            img = Image.open(f'image/{filename}')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            det = human_detector.detect(img)
            for i, (*xyxy, conf, cls) in enumerate(det):
                # xyxy -> (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
                if cls == 0 and conf >= 0.7 : # only save an image of person
                    for i in range(0, len(xyxy)):
                        if i < 2:
                            xyxy[i] -= 100.
                            if xyxy[i] < 0:
                                xyxy[i] *= -1
                        else:
                            xyxy[i] += 100.
                            if xyxy[i] > 2000:
                                xyxy[i] = 2000
                    cropedImage = crop_image(img, tuple(map(float, xyxy)))
                    cropedImage = cv2.cvtColor(np.array(cropedImage), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'actions/{filename}', cropedImage)
                    print(f"{filename} saved")
                    
if __name__ == "__main__":
    #get_image('video')
    human_detection('image')