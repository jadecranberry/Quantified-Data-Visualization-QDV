#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:24:15 2019

@author: w0
"""

import numpy as np
import pickle
import os
import math
import helper
from tqdm import tqdm


def __extract_file__(fname):
    with open(fname, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def __unflatten_image__(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img/255


def __extract_reshape_file__(fname):
    res = []
    d = __extract_file__(fname)
    images = d[b"data"]
    labels = d[b"labels"]
    for image, label in zip(images, labels):
        res.append((__unflatten_image__(image), label))
    return res


def get_images_from(dir):
    files = [f for f in os.listdir(dir) if f.startswith("data_batch")]
    res = []
    for f in files:
        res = res + __extract_reshape_file__(os.path.join(dir, f))
    return res


class Cifar(object):

    def __init__(self, dir="/content/drive/My Drive/pretrained/pretrainedCifar/data/cifar-10-batches-py/", batch_size=1):
        self.__res__ = get_images_from(dir)
        self.batch_size = batch_size
        self.batches = []
        self.__batch_num__ = 0
        for i in range(math.ceil(len(self.__res__)/batch_size)):
            self.batches.append(self.__res__[i*batch_size:(i+1)*batch_size])
        self.test_set = __extract_reshape_file__(os.path.join(dir, "test_batch"))

    def batch(self, num):
        return self.batches[num]

    def next_batch(self):
        if self.__batch_num__ <= len(self.batches):
            res = self.batches[self.__batch_num__]
            self.__batch_num__ = self.__batch_num__ + 1
        else:
            res = []

        return res

    def reset_batch(self):
        self.__batch_num__ = 0

    # The original to resize all test data all at once    
    def create_resized_test_set(self, new_size=(224, 224), dim=1000):
        self.test_set = helper.transform_to_input_output_and_pad(self.test_set, new_size=new_size, dim=dim)


    def create_test_set(self, dim=1000):
        self.test_set = helper.transform_to_input_output(self.test_set, dim=dim)

    def resize_batch_input_test(self, input_batch, new_size=(224, 224)):
        return helper.transform_to_input_and_pad(input_batch, new_size=new_size)


    # The original to resize a bacth
    # def create_resized_batches(self, new_size=(224, 224), dim=1000):
    #     self.resized_batches = [helper.transform_to_input_output_and_pad(batch, new_size=new_size, dim=dim) for batch in tqdm(self.batches, desc="Reshaping")]
    #     self.create_resized_test_set(new_size, dim)