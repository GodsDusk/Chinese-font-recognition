#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt 
import tensorflow as tf
from dataset.generator import read_text, draw_font
from nnets.vgg import vgg


def generator_images(texts, fonts):
    
    for text in texts:
        for font in fonts:
            image, _ = draw_font(text, font, mode='test')
            yield image 


def run():

    file_name = u'test.txt'
    # file_name = u'dataset/中国汉字大全.txt'
    texts = read_text(file_name)
    print len(texts)
    return
    fonts_dir = os.path.join('dataset', 'fonts')
    fonts = [os.path.join(os.getcwd(), fonts_dir, path) for path in os.listdir(fonts_dir)] 

    images_gen = generator_images(texts, fonts)

    inputs = tf.placeholder(tf.float32, shape = [None, None, 3])
    example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8) 
    example = tf.image.per_image_standardization(example)
    example = tf.expand_dims(example, 0)
    outputs = vgg(example, 2, 1.0)

    sess = tf.Session()
    restorer = tf.train.Saver()
    restorer.restore(sess, 'models/alexnet.ckpt')

    error = 0

    for index, image in enumerate(images_gen):
        
        # plt.imshow(image)
        image = np.asarray(image)
        pred = sess.run(outputs, feed_dict={inputs:image})
        pred = np.squeeze(pred)
        label = np.squeeze(np.where(pred==np.max(pred)))
        if index % 2 != label:
            error += 1

    print float(error) / index

        


if __name__ == '__main__':
    run()