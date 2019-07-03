import argparse
import numpy as np
import os
import scipy.io as sio
import cv2
import face_recognition
import sklearn
import face_preprocess

parser = argparse.ArgumentParser('Face Process')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--mark', dest='mark', help='saving name marker', type=str, default='test')
parser.add_argument('--mode', dest='mode', help='mode for test [1]verification [2]recognition [3]pair-wise comparison', type=int, default=1)
parser.add_argument('--outdata', dest='outdata', help='output data', type=str, default='aligned')

arg = parser.parse_args()

for arg_item in vars(arg):
    print('[%s] = ' % arg_item,  getattr(arg, arg_item))

                    
#face datase cleaning and processing
if arg.mode == 1:
    root_path = './'
    if not os.path.exists(arg.outdata):
        os.makedirs(arg.outdata)
    for subj in range(1,6): #totally 192 subjects starting from 1
        print('Processing Subject #%d' % subj)
        image = os.path.join(root_path, str(subj)+'.jpg')
        output = os.path.join(arg.outdata, str(subj)+'_c.jpg')
        outputg = os.path.join(arg.outdata, str(subj)+'_g.jpg')
        img = face_recognition.load_image_file(image)
        face_landmarks_list = face_recognition.face_landmarks(img)
        img_pos_table = np.zeros((5,2), dtype=np.float32)
        img_pos_table[0] = face_landmarks_list[0]['nose_bridge'][3]
        img_pos_table[1] = np.mean(face_landmarks_list[0]['left_eye'], 0)
        img_pos_table[2] = np.mean(face_landmarks_list[0]['right_eye'], 0)
        img_pos_table[3] = face_landmarks_list[0]['bottom_lip'][0]
        img_pos_table[4] = face_landmarks_list[0]['bottom_lip'][6]
        trans_image = face_preprocess.preprocess(img, None, img_pos_table, image_size="112,112")
        gray = trans_image[:,:,0]
        gray = cv2.equalizeHist(gray)
        cv2.imwrite(outputg, gray)
        cv2.imwrite(output, trans_image[:,:,::-1])
























