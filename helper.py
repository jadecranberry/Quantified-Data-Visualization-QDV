#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:18:26 2019

@author: w0
"""
# transforms [(input1, output1), (input2, output2), ...] into ([input1, input2, ...], [output1, output2, ...])
import numpy as np
import skimage


def __one_hot__(num, dim=1000):
    vec = np.zeros(dim)
    vec[num] = 1
    return vec


def transform_to_input_output(input_output, dim=1000):
    input_vals = []
    output_vals = []
    # or use  input_vals, output_vals=zip(*input_output)
    for input_val, output_val in input_output:
        input_vals.append(input_val)
        output_vals.append(output_val)

    return_value = np.array(input_vals), np.array(
        [__one_hot__(out, dim=dim)
         for out in output_vals],
        dtype="uint8")
    return return_value
  
def reshape(image, new_size):
    return skimage.transform.resize(image, new_size, mode="constant")
  
# The following is the original code to resize all test data all at once
def transform_to_input_output_and_pad(input_output, new_size=(224, 224), dim=1000):
     inp, out = transform_to_input_output(input_output, dim=dim)
     return np.array([reshape(i, new_size) for i in inp]), out

def transform_to_input_and_pad(inputs, new_size=(224, 224)):
    inp = np.array([reshape(i, new_size) for i in inputs])
    return inp


def extract_batch(batch, new_size, dim=10):
    input_batch=[]
    output_batch=[]
    for image, out in batch:
        input_batch.append(image)
        output_batch.append(__one_hot__(out, dim=dim))

    return input_batch, output_batch
  
def reshape_batch(batch, new_size, dim=10):
    input_batch=[]
    output_batch=[]
    for image, out in batch:
        new_image = reshape(image, new_size)
        input_batch.append(new_image)
        output_batch.append(__one_hot__(out, dim=dim))

    return input_batch, output_batch