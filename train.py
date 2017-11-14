#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf 
from dataset.read_data import read_data
from nnets.vgg import vgg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(batch_size):

    data_batch, annotation = read_data(batch_size)
    iterator = data_batch.make_initializable_iterator()
    inputs, outputs = iterator.get_next()
    class_num = outputs.get_shape()[-1]
    
    tf.summary.image('inputs', inputs, 16)

    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    pred = vgg(inputs, class_num, keep_prob)
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(outputs * tf.log(tf.clip_by_value(pred, 1e-5, 1.0)), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))    
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:

        writer = tf.summary.FileWriter('./log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        
        
        for i in xrange(1000):
            learning_rate = 1e-5 if i < 500 else 1e-5
            sess.run(optimizer, feed_dict={lr:learning_rate, keep_prob:0.5})
            if i % 50 == 0:
                summary, acc, l = sess.run([merged, accuracy, cross_entropy], feed_dict={keep_prob:1.0})
                print 'iter:{}, acc:{}, loss:{}'.format(i, acc, l)            

            writer.add_summary(summary, i)

        saver.save(sess, './models/vgg.ckpt')  
        
        

    return

if __name__ == '__main__':
    
    BATCH_SIZE = 128
    train(BATCH_SIZE)