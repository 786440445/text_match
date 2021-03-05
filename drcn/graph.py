import tensorflow as tf
from drcn import args
import numpy as np

class Graph:
    def __init__(self, word_embedding=None):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.p_c_index = tf.placeholder(name='p_c_index', shape=(None, args.max_char_len), dtype=tf.int32)
        self.h_c_index = tf.placeholder(name='h_c_index', shape=(None, args.max_char_len), dtype=tf.int32)
        self.p_w_index = tf.placeholder(name='p_w_index', shape=(None, args.max_word_len), dtype=tf.int32)
        self.h_w_index = tf.placeholder(name='h_w_index', shape=(None, args.max_word_len), dtype=tf.int32)
        # self.p_w_vec = tf.placeholder(name='p_w_vec', shape=(None, args.max_word_len, args.word2vec_emb_size),
                                    #   dtype=tf.float32)
        # self.h_w_vec = tf.placeholder(name='h_w_vec', shape=(None, args.max_word_len, args.word2vec_emb_size),
                                    #   dtype=tf.float32)
        self.same_word = tf.placeholder(name='same_word', shape=(None, ), dtype=tf.float32)
        self.y = tf.placeholder(name='y', shape=(None, ), dtype=tf.int32)
        self.keep_prob_embed = tf.placeholder(name='keep_prob_embed', dtype=tf.float32)
        self.keep_prob_fully = tf.placeholder(name='keep_prob_fully', dtype=tf.float32)
        self.keep_prob_ae = tf.placeholder(name='keep_prob', dtype=tf.float32)
        self.bn_training = tf.placeholder(name='bn_training', dtype=tf.bool)

        self.char_embed = tf.get_variable(name='char_embed', shape=(args.vocab_size, args.char_embedding_len),
                                          dtype=tf.float32)
        if word_embedding is not None:
            print(np.shape(word_embedding))
            print(word_embedding.dtype)
            self.word_embed = tf.get_variable("embedding_table", dtype=tf.float32, initializer=tf.constant(word_embedding), trainable=False)
        
        self.forward()

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_hidden)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_hidden)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def forward(self):
        # word representation layer
        # 字向量
        p_char_embedding = tf.nn.embedding_lookup(self.char_embed, self.p_c_index)
        h_char_embedding = tf.nn.embedding_lookup(self.char_embed, self.h_c_index)

        # 词向量
        p_word_embedding = tf.nn.embedding_lookup(self.word_embed, self.p_w_index)
        h_word_embedding = tf.nn.embedding_lookup(self.word_embed, self.h_w_index)

        same_word = tf.expand_dims(tf.expand_dims(self.same_word, axis=-1), axis=-1)
        same_word = tf.tile(same_word, [1, 20, 1])

        # [None, 20, 401]
        # 字向量+Word2vec词向量+same_word

        p = tf.concat((p_char_embedding, p_word_embedding, same_word), axis=-1)
        h = tf.concat((h_char_embedding, p_word_embedding, same_word), axis=-1)

        p = self.dropout(p, self.keep_prob_embed)
        h = self.dropout(h, self.keep_prob_embed)

        # attentively connected RNN
        for i in range(5):
            # BiLSTM
            p_state, h_state = p, h
            for j in range(5):
                with tf.variable_scope(f'p_lstm_{i}_{j}', reuse=None):
                    p_state, _ = self.BiLSTM(tf.concat(p_state, axis=-1))
                with tf.variable_scope(f'p_lstm_{i}_{j}' + str(i), reuse=None):
                    h_state, _ = self.BiLSTM(tf.concat(h_state, axis=-1))

                p_state = tf.concat(p_state, axis=-1)
                h_state = tf.concat(h_state, axis=-1)
                # attention
                cosine = tf.divide(tf.matmul(p_state, tf.matrix_transpose(h_state)),
                                   (tf.norm(p_state, axis=-1, keep_dims=True) * tf.norm(h_state, axis=-1, keep_dims=True)))
                att_matrix = tf.nn.softmax(cosine)
                p_attention = tf.matmul(att_matrix, h_state)
                h_attention = tf.matmul(att_matrix, p_state)

                # DesNet
                p = tf.concat((p, p_state, p_attention), axis=-1)
                h = tf.concat((h, h_state, h_attention), axis=-1)

            # auto_encoder
            p = tf.layers.dense(p, 256)
            h = tf.layers.dense(h, 256)

            p = self.dropout(p, self.keep_prob_ae)
            h = self.dropout(h, self.keep_prob_ae)

        # interaction and prediction layer
        add = p + h
        sub = p - h
        norm = tf.norm(sub, axis=-1)
        out = tf.concat((p, h, add, sub, tf.expand_dims(norm, axis=-1)), axis=-1)
        out = tf.reshape(out, shape=(-1, out.shape[1] * out.shape[2]))
        out = self.dropout(out, args.keep_prob_fully)

        out = tf.layers.batch_normalization(out, training=self.bn_training)
        out = tf.layers.dense(out, 256, activation='relu')
        out = tf.layers.batch_normalization(out, training=self.bn_training)
        out = tf.layers.dense(out, 256, activation='relu')
        out = tf.layers.batch_normalization(out, training=self.bn_training)
        out = tf.layers.dense(out, 384, activation='relu')
        self.logits = tf.layers.dense(out, args.class_size)
        self.train()

    def train(self):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.current_learning = tf.train.polynomial_decay(args.learning_rate, self.global_step,
                                                          decay_steps=3000, end_learning_rate=0.000001, cycle=True, power=0.5)
        self.train_op = tf.train.AdamOptimizer(self.current_learning).minimize(self.loss, global_step=self.global_step)
        # batch_normalization计算均值和方差
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([self.train_op, update_ops])
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        tf.summary.scalar('mean_loss', self.loss)
        tf.summary.scalar('acc', self.accuracy)
        self.summary = tf.summary.merge_all()


