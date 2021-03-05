import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

from drcn.graph import Graph
import tensorflow as tf
from drcn import args
from drcn.dataloader import dataloader
from utils.load_data import load_all_data
from utils.load_embedding import load_embedding_table
import numpy as np
import pickle


batch_size = 32
test_dataloader = dataloader('test.txt', batch_size=batch_size)
test_dataset = tf.data.Dataset.from_generator(test_dataloader.generator, output_types=(tf.float32, tf.int32, tf.int32, tf.int32, \
    tf.int32, tf.float32))
next_element = test_dataset.make_one_shot_iterator().get_next()

embedding_table = load_embedding_table()

model = Graph(word_embedding=embedding_table)
saver = tf.train.Saver(max_to_keep=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config)as sess:
    restore_path = os.path.join(home_dir, 'output/drcn/final_model.ckpt')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, restore_path)
    steps = int(len(test_dataloader))
    loss_all = []
    acc_all = []
    total_preds = []
    total_labels = []
    for step in range(steps):
        try:
            p_c_index_batch, h_c_index_batch, p_w_index_batch, h_w_index_batch, same_word_batch, label_batch = sess.run(next_element)
            loss, predict, acc = sess.run([model.loss, model.predict, model.accuracy],
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
            loss_all.append(loss)
            acc_all.append(acc)
            total_preds.extend(predict)
            total_labels.extend(label_batch)
        except tf.errors.OutOfRangeError:
            print('\n')

    loss = np.mean(loss_all)
    acc = np.mean(acc_all)
    f1 = calc_f1_score(total_preds, total_labels)
    print('test loss:', loss, ' test acc:', acc)
    print('f1_score:', f1)