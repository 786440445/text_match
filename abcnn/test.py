import os
import sys

home_dir = os.getcwd()
sys.path.append(home_dir)

from abcnn.graph import Graph
from abcnn import args
import tensorflow as tf
from utils.load_data import load_char_data
from tqdm import tqdm
from utils.data_utils import calc_f1_score

p, h, y = load_char_data('test.txt', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
iterator = dataset.batch(1).make_initializable_iterator()
next_element = iterator.get_next()

model = Graph(True, True)
saver = tf.train.Saver()

restore_path = os.path.join(home_dir, 'output/abcnn/final_model.ckpt')
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, restore_path)
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})

    steps = int(len(y))
    total_acc = 0
    total_loss = 0
    train_step = 0
    total_preds = []
    total_labels = []
    for step in tqdm(range(steps)):
        train_step += 1
        p_batch, h_batch, y_batch = sess.run(next_element)
        acc, preds = sess.run([model.acc, model.pred],
                       feed_dict={model.q_text: p_batch,
                                  model.a_text: h_batch,
                                  model.label: y_batch,
                                  model.keep_prob: 1})
        total_acc += acc
        total_preds.extend(preds)
        total_labels.extend(y_batch)
    f1_score = calc_f1_score(total_preds, total_labels)
    print('acc: ', total_acc / steps)
    print('f1: ', f1_score)
