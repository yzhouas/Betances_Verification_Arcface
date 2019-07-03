import face_recognition
import argparse
import numpy as np
import os
import scipy.io as sio
import cv2
import face_preprocess
import arcface
import sklearn
import glob
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Threshold to judge: 0.6 (loose) 0.55 (strict)
parser = argparse.ArgumentParser('Face Verfication Arguments')
parser.add_argument('--arcfacemodel', default='./models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--embed', dest='embed', help='[0]Dlib [1]ArcFace', type=int, default=1)
parser.add_argument('--data', dest='data', help='face data folder', type=str, default='./aligned/color')
parser.add_argument('--mode', dest='mode', help='[0]pair-wise comparison', type=int, default=0)
parser.add_argument('--thre', dest='thre', help='threshold for verification', type=float, default=0.65)

arg = parser.parse_args()

for arg_item in vars(arg):
    print('[%s] = ' % arg_item,  getattr(arg, arg_item))

def initialize_arcface(args):
    model = arcface.FaceModel(args)
    return model

#for un-ailgned faces
def compute_emb(img, file_name, mode=1, fl=0):
    #print('compute embed')
    face_locations, face_image = crop_face_with_fl(img, file_name, mode, fl)
    embed = face_recognition.face_encodings(img, face_locations)
    if len(embed)!=0 :
        return (embed[0], True)
    else:
        return (0, False)

def compute_emb_wo_crop(img, model, normalize=0):
    embed = model.get_feature(img, normalize)
    return (embed, True)

def compute_dist(emb1, emb2):
    #return np.sum(np.square(emb1-emb2),0)/512.  #cosine distance
    #return np.matmul(emb1, emb2)  #similarity
    return cosine_similarity([emb1], [emb2])
    #return cosine_distances([emb1], [emb2])

#compute pair-wise distance
if arg.mode == 0:
    accuracy_list = []
    arcface = initialize_arcface(arg)
    file_list = sorted(glob.glob(arg.data + '/*.jpg'))
    sim_table = np.zeros((len(file_list), len(file_list)))
    for i in range(len(file_list)):
        print(file_list[i])
        image_1 = cv2.imread(file_list[i])
        image_1_fe = compute_emb_wo_crop(image_1, arcface)[0]
        image_1_fe = sklearn.preprocessing.normalize([image_1_fe])
        for j in range(i+1):
            image_2 = cv2.imread(file_list[j])
            image_2_fe = compute_emb_wo_crop(image_2, arcface)[0]
            image_2_fe = sklearn.preprocessing.normalize([image_2_fe])
            sim_table[i,j] = compute_dist(image_1_fe[0], image_2_fe[0])
            sim_table[j,i] = sim_table[i,j]

    print(sim_table)


#averaging the previous four images
if arg.mode == 1:
    accuracy_list = []
    arcface = initialize_arcface(arg)
    file_list = sorted(glob.glob(arg.data + '/*.jpg'))
    sim_table = np.zeros((len(file_list), len(file_list)))
    image_gallery = []
    for i in range(len(file_list)-1):
        print(file_list[i])
        image_1 = cv2.imread(file_list[i])
        image_1_fe = compute_emb_wo_crop(image_1, arcface)[0]
        image_gallery.append(image_1_fe)
    image_gallery = np.sum(image_gallery, 0)
    image_g_fe = sklearn.preprocessing.normalize([image_gallery])
    
    image_2 = cv2.imread(file_list[-1])
    image_2_fe = compute_emb_wo_crop(image_2, arcface)[0]
    image_2_fe = sklearn.preprocessing.normalize([image_2_fe])
    similarity = compute_dist(image_g_fe[0], image_2_fe[0])

    print(similarity)



















