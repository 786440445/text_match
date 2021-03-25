import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
from undergraduate.model import Graph
from undergraduate import args
from utils.load_data import load_char_data
from undergraduate.dataloader import dataloader
from utils.load_embedding import load_embedding_table
from utils.data_utils import calc_f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_dataloader = dataloader('train.txt', batch_size=args.batch_size)
train_dataset = tf.data.Dataset.from_generator(train_dataloader.generator, output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)).repeat(args.epochs)
train_dataset = train_dataset.map(lambda x, y, m, n, z, s: (x, y, m, n, z, s), num_parallel_calls=16).prefetch(buffer_size=3000)
next_element = train_dataset.make_one_shot_iterator().get_next()

dev_dataloader = dataloader('dev.txt', batch_size=32)
dev_dataset = tf.data.Dataset.from_generator(dev_dataloader.generator, output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

embedding_table = load_embedding_table()
model = Graph(args, embedding_table)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session(config=config)as sess:
    logdir = os.path.join(home_dir, 'logs/arcnn')
    model_path = os.path.join(home_dir, 'output/mymodel')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    if args.restore:
        saver.restore(sess, os.path.join(model_path, args.restore_path))
    else:
        sess.run(tf.global_variables_initializer())
    steps = len(train_dataloader)
    old_f1 = 0
    for epoch in range(args.epochs):
        total_acc = 0
        total_loss = 0
        train_step = 0
        for step in range(steps):
            train_step += 1
            p_c_batch, h_c_batch, p_w_batch, h_w_batch, same_word_batch, y_batch = sess.run(next_element)
            batch_size = len(y_batch)
            _, loss, lr, acc, summary = sess.run([model.train_op, model.loss, model.current_learning, model.acc, model.summary],
                                    feed_dict={model.q_text: p_c_batch,
                                               model.a_text: h_c_batch,
                                               model.q_word: p_w_batch,
                                               model.a_word: h_w_batch,
                                               model.same_word: same_word_batch,
                                               model.label: y_batch,
                                               model.batch_size: batch_size,
                                               model.is_training: True,
                                               model.keep_prob: args.keep_prob})
            total_loss += loss
            total_acc += acc
            print('epoch: %d   step: %d   loss: %.4f   lr: %.6f    acc: %.4f' % (epoch, step,
                                                                                 total_loss/train_step,
                                                                                 lr,
                                                                                 total_acc/train_step))

        eval_steps = len(dev_dataloader)
        next_element_eval = dev_dataset.make_one_shot_iterator().get_next()
        total_loss = 0
        total_acc = 0
        total_preds = []
        total_labels = []
        for step in range(eval_steps):
            p_batch, h_batch, p_w_batch, h_w_batch, same_word_batch, y_batch = sess.run(next_element_eval)
            batch_size = len(y_batch)
            loss_eval, acc_eval, preds = sess.run([model.loss, model.acc, model.prediction],
                                           feed_dict={model.q_text: p_batch,
                                                      model.a_text: h_batch,
                                                      model.q_word: p_w_batch,
                                                      model.a_word: h_w_batch,
                                                      model.same_word: same_word_batch,
                                                      model.label: y_batch,
                                                      model.batch_size: batch_size,
                                                      model.is_training: False,
                                                      model.keep_prob: 1})
            total_loss += loss_eval
            total_acc += acc_eval
            total_preds.extend(preds)
            total_labels.extend(y_batch)
        f1_score = calc_f1_score(total_preds, total_labels)
        acc = total_acc/eval_steps
        print('loss_eval: %.4f  f1: %.4f    acc_eval: %.4f' % (total_loss/eval_steps, f1_score, acc))
        if f1_score > old_f1:
            saver.save(sess, os.path.join(model_path, 'SE-ACNN_{}_f1_{:.4f}_acc_{:.4f}.ckpt'.format(epoch, f1_score, acc)))
            saver.save(sess, os.path.join(model_path, 'final_model.ckpt'))
            old_f1 = f1_score
        else:
            print('not improved')