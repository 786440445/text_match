import os
import sys

home_dir = os.getcwd()
sys.path.append(home_dir)

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np

# 是否能成功加载GPU
from text_match.apcnn.graph import Graph
from text_match.apcnn import args
from text_match.utils.load_data import load_char_data

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

p, h, y = load_char_data('train.txt', data_size=None)
p_eval, h_eval, y_eval = load_char_data('dev.txt', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
iterator = dataset.batch(args.batch_size).repeat(args.epochs).make_initializable_iterator()
next_element = iterator.get_next()

dataset_eval = dataset.batch(args.batch_size)
iterator_eval = dataset_eval.make_initializable_iterator()
next_element_eval = iterator_eval.get_next()

model = Graph()
saver = tf.train.Saver(max_to_keep=5)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    logdir = os.path.join(home_dir, 'text_match/logs/apcnn')
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})
    steps = int(len(y) / args.batch_size)
    old_eval_acc = 0
    for epoch in range(args.epochs):
        model.is_training = True
        total_acc = 0
        total_loss = 0
        train_step = 0
        for step in range(steps):
            train_step += 1
            p_batch, h_batch, y_batch = sess.run(next_element)
            batch_size = len(p_batch)
            _, loss, all_loss, acc, lr, summary = sess.run([model.train_op, model.loss, model.all_loss, model.acc, model.current_learning,
                                                  model.summary],
                                    feed_dict={model.q_text: p_batch,
                                               model.a_text: h_batch,
                                               model.label: y_batch,
                                               model.batch_size: batch_size,
                                               model.keep_prob: args.keep_prob})
            total_loss += loss
            total_acc += acc
            print('epoch: %d   step: %d   loss: %.4f    lr: %.6f    acc: %.4f' % (epoch, step,
                                                                      total_loss/train_step, lr, total_acc/train_step))
            if loss == np.nan:
                print(all_loss)
        writer.add_summary(summary, epoch)
        eval_steps = int(len(y_eval) / args.batch_size)
        model.is_training = False
        total_loss = 0
        total_acc = 0
        sess.run(iterator_eval.initializer, feed_dict={p_holder: p_eval, h_holder: h_eval, y_holder: y_eval})
        next_element_eval = iterator_eval.get_next()
        for step in range(eval_steps):
            p_batch, h_batch, y_batch = sess.run(next_element_eval)
            batch_size = len(p_batch)
            loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                           feed_dict={model.q_text: p_batch,
                                                      model.a_text: h_batch,
                                                      model.label: y_batch,
                                                      model.batch_size: batch_size,
                                                      model.keep_prob: 1})
            total_loss += loss_eval
            total_acc += acc_eval
        print('loss_eval: %.4f  acc_eval: %.4f' % (total_loss/eval_steps, total_acc/eval_steps))
        saver.save(sess, os.path.join(home_dir, 'text_match/output/apcnn/apcnn_{}_{:.4f}.ckpt'.format(epoch, total_acc/eval_steps)))
        if acc_eval > old_eval_acc:
            saver.save(sess, os.path.join(home_dir, 'text_match/output/apcnn/final_model.ckpt'))
            old_eval_acc = acc_eval
        else:
            print('not improved')