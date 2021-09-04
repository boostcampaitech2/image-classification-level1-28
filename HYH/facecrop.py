import torch
import pandas as pd
import numpy as np
from facenet_pytorch import MTCNN

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True)

def faceCrop(img, output_img_size : tuple = (310,310)): 
    '''
        image의 얼굴부분만 Crop 하는 함수
        img(ndarray) : cv2.imread와 cv2.cvtColor(,cv2.COLOR_BGR2RGB)의 결과값 입력
        output_img_size : output으로 받고싶은 image size 입력
    '''
    
    boxes,probs = mtcnn.detect(img) # MTCNN을 활용하여 img내에서 얼굴을 찾음
                                    # 찾은 경우 -> boxes : 좌표값, probs : 확률값 출력
                                    # 못찾은 경우 -> boxes : None, probs : [None] 출력
    
    if probs[0] == None: # 못찾은 경우, image center 값 반환
        return img_center_crop(img, output_img_size)
    
    else :  # 찾은 경우, 얼굴 center기준으로 outputsize를 계산하여 반환    
        xmid = int((boxes[0, 0]+boxes[0, 2])/2)
        ymid = int((boxes[0, 1]+boxes[0, 3])/2)

        xmin = xmid - output_img_size[0]//2
        ymin = ymid - output_img_size[1]//2
        xmax = xmid + output_img_size[0] - output_img_size[0]//2
        ymax = ymid + output_img_size[1] - output_img_size[1]//2
        
        if any((xmin < 0,xmax > img.shape[1],ymin < 0,ymax > img.shape[0])): # 찾은 후 box가 image 사이즈보다 클 경우 center 위치 반환
            return img_center_crop(img, output_img_size)
        
        else :  #정상적으로 얼굴만 crop된 image 반환
            new_img = img[ymin:ymax, xmin:xmax, :]
            return new_img

def img_center_crop(img, output_img_size : tuple = (100,100)): #image 내에서 얼굴을 못찾거나, image를 벗어난 경우 center crop 하는 함수
    ximg_cen = int((img.shape[1])/2)
    yimg_cen = int((img.shape[0])/2)

    xmin = ximg_cen - output_img_size[0]//2
    ymin = yimg_cen - output_img_size[1]//2
    xmax = ximg_cen + output_img_size[0] - output_img_size[0]//2
    ymax = yimg_cen + output_img_size[1] - output_img_size[1]//2   
    new_img = img[ymin:ymax, xmin:xmax, :]
    
    return new_img