import os
import sys

home_dir = os.getcwd()
sys.path.append(home_dir)

from dssm.graph import Graph
from dssm import args
import tensorflow as tf
from utils.load_data import load_char_data
from tqdm import tqdm

p, h, y = load_char_data('test.txt', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
iterator = dataset.batch(16).make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

restore_path = os.path.join(home_dir, 'output/dssm/final_model.ckpt')
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, restore_path)
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})

    steps = int(len(y) / 16)
    total_acc = 0
    total_loss = 0
    train_step = 0
    for step in tqdm(range(steps)):
        train_step += 1
        p_batch, h_batch, y_batch = sess.run(next_element)
        acc = sess.run(model.acc,
                       feed_dict={model.p: p_batch,
                                  model.h: h_batch,
                                  model.y: y_batch,
                                  model.keep_prob: 1})
        total_acc += acc
    print('acc: ', total_acc / steps)
