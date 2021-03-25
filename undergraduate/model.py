#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nlp_tools -> model
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/3 9:14 PM
@Desc   ：
=================================================='''
from undergraduate.args import keep_prob
import tensorflow as tf


class Graph:
    def __init__(self, args, word_embedding=None):
        self.args = args
        self.embedding_size = args.embedding_size
        # self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.word_length = args.word_length
        self.vocab_size = args.vocab_size
        self.class_size = args.class_size
        self.learning_rate = args.learning_rate

        self.word_embedding = word_embedding

        self.filter_nums = [(3, 32), (4, 32), (5, 32)]
        self.initializer = None

        self.init_placeholder()
        self.forward()

    def init_placeholder(self):
        self.q_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.a_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.q_word = tf.placeholder(dtype=tf.int32, shape=[None, self.word_length])
        self.a_word = tf.placeholder(dtype=tf.int32, shape=[None, self.word_length])
        self.label = tf.placeholder(dtype=tf.int32, shape=None)
        self.same_word = tf.placeholder(name='same_word', shape=(None, ), dtype=tf.float32)
        self.is_training = tf.placeholder(name='bn_training', dtype=tf.bool)
        
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.embedding_table = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size),
                                               name='embedding')
        if self.word_embedding is not None:
            self.word_embed = tf.get_variable("embedding_table", dtype=tf.float32,
             initializer=tf.constant(self.word_embedding), trainable=False)
    
    def similarity_model(self, output_a, output_b):
        def model_func(input_a, input_b, scope):
            with tf.variable_scope(scope):
                input_concat = tf.concat((input_a, input_b, tf.multiply(input_a, input_b), input_a - input_b), axis=-1)
                output = tf.layers.dense(input_concat, 32, activation=tf.nn.tanh, 
                    use_bias=True, kernel_initializer=self.initializer)
            return output

        def build_gate(input_a, input_b, scope):
            with tf.variable_scope(scope):
                input_concat = tf.concat((input_a, input_b), axis=-1)
                gate = tf.layers.dense(input_concat, 32, activation=tf.nn.sigmoid, 
                    use_bias=False, kernel_initializer=self.initializer)
            return gate

        # (b, s)
        m_oa_ob = model_func(output_a, output_b, scope='m_a')
        m_ob_oa = model_func(output_b, output_a, scope='m_b')

        # State Function
        gate_a = build_gate(output_a, output_b, scope='gate_a')
        gate_b = build_gate(output_b, output_a, scope='gate_b')
        # [B, 64]
        output_a = tf.multiply(gate_a, m_oa_ob) + (1 - gate_a) * output_a
        output_b = tf.multiply(gate_b, m_ob_oa) + (1 - gate_b) * output_b
        output_concat = tf.concat((output_a, output_b), axis=-1)
        return output_concat

    def forward(self):
        same_word = tf.expand_dims(tf.expand_dims(self.same_word, axis=-1), axis=-1)
        same_word = tf.tile(same_word, [1, self.args.max_length, 1])

        self.q_embedding = tf.nn.embedding_lookup(self.embedding_table, self.q_text)
        self.a_embedding = tf.nn.embedding_lookup(self.embedding_table, self.a_text)
        self.word_q_embedding = tf.nn.embedding_lookup(self.word_embed, self.q_word)
        self.word_a_embedding = tf.nn.embedding_lookup(self.word_embed, self.a_word)

        self.query_in = tf.concat((self.q_embedding, self.word_q_embedding, same_word), -1)
        self.answer_in = tf.concat((self.a_embedding, self.word_a_embedding, same_word), -1)
        
        # [B, 20, 401]
        self.query_in = self.dense(self.query_in,  128, activation='swish', keep_prob=0.8)
        self.answer_in = self.dense(self.answer_in, 128, activation='swish', keep_prob=0.8)

        self.querys, self.answers = self.encode(self.query_in, self.answer_in)

        self.query_cnn_enc = tf.concat(self.querys, axis=-1)
        self.answer_cnn_enc = tf.concat(self.answers, axis=-1)
        
        output_a = self.dense(self.query_cnn_enc, 128, activation='swish', keep_prob=0.8)
        output_b = self.dense(self.answer_cnn_enc, 128, activation='swish', keep_prob=0.8)

        self.concat = tf.concat((output_a, output_b, output_a - output_b, tf.abs(output_a - output_b), output_a + output_b), axis=-1)
        # self.concat = tf.concat((output_a, output_b), axis=-1)
        # print('output_a', output_a)
        # output_a = tf.contrib.layers.layer_norm(output_a)
        # output_b = tf.contrib.layers.layer_norm(output_b)
        # output_a = self.dense(output_a, 32, activation='relu')
        # output_b = self.dense(output_b, 32, activation='relu')
        # self.concat = self.similarity_model(output_a, output_b)
        # self.score = self._cosine(self.query, self.answer)
        # self._model_stats()  # print model statistics info
        # self.score = tf.expand_dims(self.score, -1)
        # neg_result = 1 - self.score
        # self.cos_logits = tf.concat([neg_result, self.score], axis=1)
        shape = self.concat.shape.as_list()
        logits = tf.reshape(self.concat, [-1, shape[1]*shape[2]])
        logits = self.batch_norm(logits)
        logits = self.dense(logits, 384, activation='swish', keep_prob=0.8)
        logits = self.batch_norm(logits)
        logits = self.dense(logits, 256, activation='swish', keep_prob=0.8)
        logits = self.dense(logits, 2, activation='softmax')
        self.train(logits)

    def BiLSTM(self, x, layer):
        if layer == 1:
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        else: 
            fw_cell = [tf.nn.rnn_cell.BasicLSTMCell(128) for _ in range(layer)]
            bw_cell = [tf.nn.rnn_cell.BasicLSTMCell(128) for _ in range(layer)]
            multi_fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell)
            multi_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell)
            return tf.nn.bidirectional_dynamic_rnn(multi_fw_cell, multi_bw_cell, x, dtype=tf.float32)
    
    def BiGRU(self, x):
        fw_cell = tf.nn.rnn_cell.GRUCell(64)
        bw_cell = tf.nn.rnn_cell.GRUCell(64)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def encode(self, query, answer):
        query_result = []
        answer_result = []
        # [B, L, 128]
        for layer, (size, filters) in enumerate(self.filter_nums):
            query1 = tf.expand_dims(query, -1)
            answer1 = tf.expand_dims(answer, -1)
            # [B, L, 128, 1]
            query1 = self.cnn_cell(query1, filters, size) # [B, 20, 128, 32]
            query2 = self.nin_network(query1, filters/4)

            answer1 = self.cnn_cell(answer1, filters, size)
            answer2 = self.nin_network(answer1, filters/4)

            query1 += self.squeeze_excitation_layer(
                self.cnn_cell(query2, filters, size), out_dim=filters, ratio=4)
            answer1 += self.squeeze_excitation_layer(
                self.cnn_cell(answer2, filters, size), out_dim=filters, ratio=4)

            shape = query1.shape.as_list()
            query1 = tf.reshape(query1, shape=[-1, shape[1], shape[2] * shape[3]])
            answer1 = tf.reshape(answer1, shape=[-1, shape[1], shape[2] * shape[3]])

            query1 = self.dense(query1, 128, 'swish')
            answer1 = self.dense(answer1, 128, 'swish')

            with tf.variable_scope('q_lstm_%s' % layer, reuse=None):
                self.query_pre, _ = self.BiLSTM(query1, 1)
            with tf.variable_scope('a_lstm_%s' % layer, reuse=None):
                self.answer_pre, _ = self.BiLSTM(answer1, 1)

            query2 = tf.concat(self.query_pre, -1)
            query2 = self.dense(query2, 128, 'swish')
            
            answer2 = tf.concat(self.answer_pre, -1)
            answer2 = self.dense(answer2, 128, 'swish')

            attentionQ = self.AttentionLayer(query2, answer2)
            attentionA = self.AttentionLayer(answer2, query2)

            attention_q, attention_a = self.attention_pooling(query2, answer2)
            V_q = tf.concat((query2, attentionQ, attention_q, query2 - attention_q, tf.multiply(query2, attention_q)), axis=-1)
            V_a = tf.concat((answer2, attentionA, attention_a, answer2 - attention_a, tf.multiply(answer2, attention_a)), axis=-1)

            V_q = self.dense(V_q, 256, activation='swish', keep_prob=0.8)
            V_a = self.dense(V_a, 256, activation='swish', keep_prob=0.8)

            query_result.append(V_q)
            answer_result.append(V_a)

        return query_result, answer_result


    def dropout(self, x, keep_prob):
        return tf.layers.dropout(x, 1-keep_prob, training=self.is_training)

    def cnn_cell(self, x, filters, size, pool=False):
        x = self.batch_norm(self.conv2d(x, filters, size))
        if pool:
            x = self.maxpool(x)
        return x
    
    def nin_network(self, x, size):
        return self.batch_norm(self.conv1x1(x, size))
        
    def conv1x1(self, input, filters):
        return tf.layers.conv2d(input, filters=filters, kernel_size=(1, 1),
                                use_bias=True, activation='relu',
                                padding='same', kernel_initializer=self.initializer)

    def conv2d(self, inputs, filters, size):
        return tf.layers.conv2d(inputs, filters=filters, kernel_size=(size, size),
                                use_bias=True, activation=tf.nn.relu,
                                padding='same', kernel_initializer=self.initializer)
    
    def conv1d(self, inputs, filters, size):
        return tf.layers.conv1d(inputs, filters=filters, kernel_size=(size),
                                use_bias=True, activation=tf.nn.relu,
                                padding='same', kernel_initializer=self.initializer)

    def maxpool(self, inputs):
        return tf.layers.max_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2), padding='valid')

    def averagepool(self, inputs):
        return tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2), padding='valid')

    def dense(self, inputs, units, activation, keep_prob=None):
        if activation == 'relu':
            ret = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                                   use_bias=True, kernel_initializer=self.initializer)
        elif activation == 'leaky_relu':
            ret = tf.layers.dense(inputs, units, activation=tf.nn.leaky_relu,
                                   use_bias=True, kernel_initializer=self.initializer)
        elif activation == 'selu':
            ret = tf.layers.dense(inputs, units, activation=tf.nn.selu,
                                   use_bias=True, kernel_initializer=self.initializer)
        elif activation == 'swish':
            ret = tf.layers.dense(inputs, units, activation=tf.nn.swish, use_bias=True, kernel_initializer=self.initializer)
        else:
            ret = tf.layers.dense(inputs, units, activation=activation,
                                   use_bias=True, kernel_initializer=self.initializer)

        if keep_prob:
            return self.dropout(ret, keep_prob)
        else:
            return ret

    def batch_norm(self, input_x):
        return tf.layers.batch_normalization(input_x, training=self.is_training)

    def global_average_pooling(self, inputs):
        return tf.reduce_mean(inputs, [1], keep_dims=True)
    
    def global_max_pooling(self, inputs):
        return tf.reduce_max(inputs, [1], keep_dims=True)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio):
        input_x = self.batch_norm(input_x)
        squeeze = self.global_average_pooling(input_x)
        excitation = self.dense(squeeze, units=out_dim / ratio, activation='relu')
        excitation = self.dense(excitation, units=out_dim, activation='sigmoid')
        scale = tf.multiply(input_x, excitation)
        return scale

    def AttentionLayer(self, query, value):
        # Q.Vt
        attn_scores = tf.matmul(query, value, transpose_b=True)
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        context = tf.matmul(attn_probs, value)
        return context

    def attention_pooling(self, q, a):
        hidden_size = q.get_shape().as_list()[-1]  # vector size
        with tf.variable_scope('attention_pooling', reuse=tf.AUTO_REUSE):
            self.Q = q
            self.A = a
            self.U = tf.get_variable('U', [hidden_size, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.U_batch = tf.tile(tf.expand_dims(self.U, 0), [self.batch_size, 1, 1])
            
            self.G = tf.tanh(
                tf.matmul(
                    tf.matmul(self.Q, self.U_batch), tf.transpose(self.A, [0, 2, 1]))
            )  # G    b*m*n
            self.sigma_q = tf.nn.softmax(self.G, axis=2)
            self.sigma_a = tf.nn.softmax(self.G, axis=1)
            # self.sigma_q2 = tf.nn.softmax(self.mean_g_q, axis=1)
            # self.sigma_a2 = tf.nn.softmax(self.mean_g_a, axis=2)
            # self.mean_g_q = tf.reduce_mean(self.sigma_q, axis=2, keepdims=True)
            # self.max_g_q = tf.reduce_max(self.sigma_q, axis=2, keepdims=True)

            # self.mean_g_a = tf.reduce_mean(self.sigma_a, axis=1, keepdims=True)
            # self.max_g_a = tf.reduce_max(self.sigma_a, axis=1, keepdims=True)
            
            # self.sigma_q1 = self.mean_g_q
            # self.sigma_q2 = self.max_g_q
            # self.sigma_a1 = self.mean_g_a
            # self.sigma_a2 = self.max_g_a

            # attention_max_a = tf.matmul(self.sigma_a1, self.A)
            # attention_mean_a = tf.matmul(self.sigma_a2, self.A)
            # attention_max_q = tf.transpose(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), self.sigma_q1), [0, 2, 1])
            # attention_mean_q = tf.transpose(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), self.sigma_q2), [0, 2, 1])
            # [B, 1, 64]
            # r_q = tf.squeeze(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), self.sigma_q), axis=2)
            # r_a = tf.squeeze(tf.matmul(self.sigma_a, self.A), axis=1)
            # [L2, L1] * [L1, D] = [L2, D]
            self.attention_a = tf.matmul(tf.transpose(self.sigma_a, [0, 2, 1]), self.Q)
            # [L1, L2] * [L2. D] = [L1, D]
            self.attention_q = tf.matmul(self.sigma_q, self.A)
            return self.attention_q, self.attention_a

    def train(self, logits):
        y = tf.one_hot(self.label, self.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)

        self.current_learning = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                          decay_steps=5000, end_learning_rate=0.000001, cycle=True,
                                                          power=0.5)
        self.train_op = tf.train.AdamOptimizer(self.current_learning).minimize(self.loss, global_step=self.global_step)

        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.group([self.train_op, update_ops])

        self.prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.label)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('mean_loss', self.loss)
        tf.summary.scalar('acc', self.acc)
        self.summary = tf.summary.merge_all()

    @staticmethod
    def _cosine(x, y):
        """x, y shape (batch_size, vector_size)"""
        cosine = tf.div(
            tf.reduce_sum(x * y, 1),
            tf.sqrt(tf.reduce_sum(x * x, 1)) * tf.sqrt(tf.reduce_sum(y * y, 1)) + 1e-8,
            name="cosine")
        return cosine

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return tf.reduce_all(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))