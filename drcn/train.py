import os
import sys

home_dir = os.getcwd()
sys.path.append(home_dir)

import numpy as np
from drcn.graph import Graph
import tensorflow as tf
from drcn import args
from drcn.dataloader import dataloader
from utils.load_data import load_all_data
from utils.load_embedding import load_embedding_table
from utils.data_utils import calc_f1_score
import pickle


train_dataloader = dataloader('train.txt', batch_size=args.batch_size)
train_dataset = tf.data.Dataset.from_generator(train_dataloader.generator, output_types=(tf.int32, tf.int32, tf.int32, tf.int32, \
    tf.int32, tf.int32))
train_dataset = train_dataset.map(lambda x, y, z, w, p, q: (x, y, z, w, p, q), num_parallel_calls=16).prefetch(buffer_size=3000)

dev_dataloader = dataloader('dev.txt', batch_size=10)
dev_dataset = tf.data.Dataset.from_generator(dev_dataloader.generator, output_types=(tf.int32, tf.int32, tf.int32, tf.int32, \
    tf.int32, tf.int32))
dev_dataset = dev_dataset.map(lambda x, y, z, w, p, q: (x, y, z, w, p, q), num_parallel_calls=8).prefetch(buffer_size=1000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.80

embedding_table = load_embedding_table()

model = Graph(embedding_table)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session(config=config)as sess:
    logdir = os.path.join(home_dir, 'logs/drcn')
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    if args.restore:
        saver.restore(sess, os.path.join('output/drcn', args.restore_path))
    else:
        sess.run(tf.global_variables_initializer())
    steps = len(train_dataloader)
    print('steps : ', steps)
    old_f1 = 0
    all_step = 0
    for epoch in range(args.epochs):
        total_loss = 0
        total_acc = 0
        next_element = train_dataset.make_one_shot_iterator().get_next()
        for step in range(steps):
            try:
                p_c_index_batch, h_c_index_batch, p_w_index_batch, h_w_index_batch, same_word_batch, label_batch = sess.run(
                    next_element)
                loss, _, predict, acc, lr, summary = sess.run([model.loss, model.train_op, model.predict, model.accuracy, model.current_learning, model.summary],
                                                 feed_dict={model.p_c_index: p_c_index_batch,
                                                            model.h_c_index: h_c_index_batch,
                                                            model.p_w_index: p_w_index_batch,
                                                            model.h_w_index: h_w_index_batch,
                                                            model.same_word: same_word_batch,
                                                            model.y: label_batch,
                                                            model.keep_prob_embed: args.keep_prob_embed,
                                                            model.keep_prob_fully: args.keep_prob_fully,
                                                            model.keep_prob_ae: args.keep_prob_ae,
                                                            model.bn_training: True})
                total_acc += acc
                total_loss += loss
                all_step += 1
                if all_step % 50 == 0:
                    writer.add_summary(summary, epoch * steps + step)
                if all_step % 5 == 0:
                    print('epoch: {}    step: {}    lr: {:.6f}  average_loss: {:.6f}    average_acc: {:.6f}'.format(epoch, step, lr, total_loss/(step+1), total_acc/(step+1)))

            except tf.errors.OutOfRangeError:
                print('\n')
        dev_batchs = len(dev_dataloader)
        total_loss = 0
        total_acc = 0
        total_preds = []
        total_labels = []
        dev_next_element = dev_dataset.make_one_shot_iterator().get_next()
        for step in range(dev_batchs):
            p_c_index_batch, h_c_index_batch, p_w_index_batch, h_w_index_batch, same_word_batch, label_batch = sess.run(
                dev_next_element)
            predict, acc, loss = sess.run([model.predict, model.accuracy, model.loss],
                                          feed_dict={model.p_c_index: p_c_index_batch,
                                                     model.h_c_index: h_c_index_batch,
                                                     model.p_w_index: p_w_index_batch,
                                                     model.h_w_index: h_w_index_batch,
                                                     model.same_word: same_word_batch,
                                                     model.y: label_batch,
                                                     model.keep_prob_embed: 1,
                                                     model.keep_prob_fully: 1,
                                                     model.keep_prob_ae: 1,
                                                     model.bn_training: False})
            total_acc += acc
            total_loss += loss
            total_preds.extend(predict)
            total_labels.extend(label_batch)

        dev_loss = total_loss / dev_batchs
        dev_acc = total_acc / dev_batchs
        eval_f1 = calc_f1_score(total_preds, total_labels)
        print('epoch: {}    average_loss: {:.4f}    eval_f1: {:.4f} eval_acc: {:.4f}'.format(epoch, dev_loss, eval_f1, dev_acc))
        if eval_f1 > old_f1:
            save_path = os.path.join(home_dir, args.save_model_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            saver.save(sess, os.path.join(save_path, 'drcn_{}_f1_{:.4f}_acc_{:.4f}.ckpt'.format(epoch, eval_f1, dev_acc)))
            saver.save(sess, os.path.join(save_path, 'final_model.ckpt'))
            print('save model done')
            old_f1 = eval_f1
        else:
            print('not improved')
