from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
xrange=range
from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, args):
    self.args = args
    model = edict()

    self.det_minsize = 50
    self.det_threshold = [0.4,0.6,0.6]
    self.det_factor = 0.9
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.image_size = image_size
    _vec = args.arcfacemodel.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)

    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model

 
  def get_feature(self, face_img, normalize=1):
    #face_img is bgr image just loaded from cv2 function
    nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.args.flip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    if normalize == 1:
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
    elif normalize == 0:
        embedding = embedding.flatten()
    return embedding
   
  
  def get_ensemble_feature(self, face_img_list, normalize=1):
    #face_img is bgr image just loaded from cv2 function
    embedding = None

    for i in range(len(face_img_list)):
      face_img = face_img_list[i]
      nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
    
      for flipid in [0,1]:
        if flipid==1:
          if self.args.flip==0:
            break
        do_flip(aligned)
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        _embedding = self.model.get_outputs()[0].asnumpy()
        #print(_embedding.shape)
        if embedding is None:
          embedding = _embedding
        else:
          embedding += _embedding
    if normalize==1:
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
    elif normalize==0:
        embedding = embedding.flatten()
    return embedding
