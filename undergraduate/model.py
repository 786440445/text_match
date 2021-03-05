#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nlp_tools -> model
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/3 9:14 PM
@Desc   ：
=================================================='''
import tensorflow as tf


class Graph:
    def __init__(self, args, word_embedding=None):
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
        print('m_oa_ob', m_oa_ob)
        print('output_a', output_a)
        print('gate_a', gate_a)
        # [B, 64]
        output_a = tf.multiply(gate_a, m_oa_ob) + (1 - gate_a) * output_a
        output_b = tf.multiply(gate_b, m_ob_oa) + (1 - gate_b) * output_b
        output_concat = tf.concat((output_a, output_b), axis=-1)
        return output_concat

    def forward(self):
        same_word = tf.expand_dims(tf.expand_dims(self.same_word, axis=-1), axis=-1)
        same_word = tf.tile(same_word, [1, 20, 1])

        self.q_embedding = tf.nn.embedding_lookup(self.embedding_table, self.q_text)
        self.a_embedding = tf.nn.embedding_lookup(self.embedding_table, self.a_text)
        self.word_q_embedding = tf.nn.embedding_lookup(self.word_embed, self.q_word)
        self.word_a_embedding = tf.nn.embedding_lookup(self.word_embed, self.a_word)

        self.query_in = tf.concat((self.q_embedding, self.word_q_embedding, same_word), -1)
        self.answer_in = tf.concat((self.a_embedding, self.word_a_embedding, same_word), -1)
        
        # [B, 20, 401]
        self.query_in = self.dense(self.query_in,  64, activation='relu')
        self.query_in = self.dropout(self.query_in, 0.8)
        self.answer_in = self.dense(self.answer_in, 64, activation='relu')
        self.answer_in = self.dropout(self.answer_in, 0.8)

        # [B, 20, 128]
        self.query_in = tf.contrib.layers.layer_norm(self.query_in)
        self.answer_in = tf.contrib.layers.layer_norm(self.answer_in)

        print(self.query_in)
        with tf.variable_scope('p_lstm', reuse=None):
            self.query_pre, _ = self.BiLSTM(self.query_in)
        with tf.variable_scope('a_lstm', reuse=None):
            self.answer_pre, _ = self.BiLSTM(self.answer_in)

        self.query_pre = tf.concat(self.query_pre, -1)
        self.answer_pre = tf.concat(self.answer_pre, -1)
        # self.query_pre = tf.contrib.layers.layer_norm(self.query_pre)
        # self.answer_pre = tf.contrib.layers.layer_norm(self.answer_pre)
        self.query_pre = self.dense(self.query_pre, 64, activation='relu')
        self.answer_pre = self.dense(self.answer_pre, 64, activation='relu')

        self.querys, self.answers = self.encode(self.query_pre, self.answer_pre)
        self.querys.append(self.query_pre)
        self.answers.append(self.answer_pre)
        self.query_cnn_enc = tf.concat(self.querys, axis=-1)
        self.answer_cnn_enc = tf.concat(self.answers, axis=-1)
        
        print('query_cnn_enc: ', self.query_cnn_enc)
        print('answer_cnn_enc: ', self.answer_cnn_enc)

        self.query_cnn_enc = self.dense(self.query_cnn_enc, 64, activation='relu')
        self.query_cnn_enc = self.dropout(self.query_cnn_enc, 0.8)

        self.answer_cnn_enc = self.dense(self.answer_cnn_enc, 64, activation='relu')
        self.answer_cnn_enc = self.dropout(self.answer_cnn_enc, 0.8)

        self.average_cnn_q = self.global_average_pooling(self.query_cnn_enc)
        self.max_cnn_q = self.global_max_pooling(self.query_cnn_enc)
        # print('average_cnn_q', self.average_cnn_q)
        # print('max_cnn_q', self.max_cnn_q)

        self.average_cnn_a = self.global_average_pooling(self.answer_cnn_enc)
        self.max_cnn_a = self.global_max_pooling(self.answer_cnn_enc)

        self.q_enc = tf.concat((tf.squeeze(self.average_cnn_q, 1), tf.squeeze(self.max_cnn_q, 1)), -1)
        self.a_enc = tf.concat((tf.squeeze(self.average_cnn_a, 1), tf.squeeze(self.max_cnn_a, 1)), -1)
        # print('encode_cnn_a', self.encode_cnn_a)

        self.attention_q, self.attention_a = self.attention_pooling(self.query_pre, self.answer_pre)
        print('attention_q ----', self.attention_q)
        # (?, 20, 64)
        print('attention_a ----', self.attention_a)
        # (?, 20, 64)
        V_a = tf.concat((self.query_cnn_enc, self.attention_q, self.query_cnn_enc - self.attention_q, tf.multiply(self.query_cnn_enc, self.attention_q)), axis=-1)
        V_b = tf.concat((self.answer_cnn_enc, self.attention_a, self.answer_cnn_enc - self.attention_a, tf.multiply(self.answer_cnn_enc, self.attention_a)), axis=-1)
        print('V_a', V_a)
        print('V_b', V_b)
        v_a_max = tf.reduce_max(V_a, axis=1)
        v_a_avg = tf.reduce_mean(V_a, axis=1)
        v_b_max = tf.reduce_max(V_b, axis=1)
        v_b_avg = tf.reduce_mean(V_b, axis=1)
        print('v_a_max', v_a_max)
        # 64 + 64 + 64
        output_a = tf.concat((v_a_max, self.q_enc, v_a_avg), axis=-1)
        # (8*s_b -8)
        output_b = tf.concat((v_b_max, self.a_enc, v_b_avg), axis=-1)
        # [256]
        # print('attention', self.query)
        # print('attention', self.answer)
        # self.query = tf.squeeze(self.query, 1)                
        # self.answer = tf.squeeze(self.answer, 1)
        
        self.concat = tf.concat((output_a, output_b, tf.abs(output_a - output_b), output_a + output_b), axis=-1)
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

        # shape = self.concat.shape.as_list()
        # self.logits = tf.reshape(self.concat, [-1, shape[1]*shape[2]])
        self.logits = self.dense(self.concat, 64, activation=tf.nn.tanh)
        self.logits = self.dense(self.logits, 2, activation='softmax')
        self.train(self.logits)

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
    
    def BiGRU(self, x):
        fw_cell = tf.nn.rnn_cell.GRUCell(64)
        bw_cell = tf.nn.rnn_cell.GRUCell(64)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def encode(self, query, answer):
        query_result = []
        answer_result = []
        print('encoder query in:', query)
        print('encoder answer in:', answer)
        # [B, L, 128]
        for layer, (size, filters) in enumerate(self.filter_nums):
            query1 = tf.expand_dims(query, -1)
            answer1 = tf.expand_dims(answer, -1)
            # [B, L, 128, 1]
            query1 = self.nin_network(query1, filters/4)
            query1 = self.cnn_cell(query1, filters, size) # [B, 20, 128, 32]
            # print('query1', query1)
            # print('query1', query1)
            # [B, L, 128, 8]
            answer1 = self.nin_network(answer1, filters/4)
            answer1 = self.cnn_cell(answer1, filters, size)

            shape = query1.shape.as_list()
            # [B, L 128 * 8]
            query2 = tf.reshape(query1, [-1, shape[1], shape[2]*shape[3]])
            answer2 = tf.reshape(answer1, [-1, shape[1], shape[2]*shape[3]])
            # print('query2', query2)
            # [B, L, 1024]
            with tf.variable_scope('p_gru_' + str(layer), reuse=None):
                self.q_state, _ = self.BiGRU(query2)
            with tf.variable_scope('a_gru_' + str(layer), reuse=None):
                self.a_state, _ = self.BiGRU(answer2)
            q_state = tf.concat(self.q_state, -1)
            a_state = tf.concat(self.a_state, -1)
            # print('q_state: ', q_state)
            # # [B, L, 128]
            q_state = self.dense(q_state, filters, activation='relu')
            a_state = self.dense(a_state, filters, activation='relu')
            
            query1 += self.squeeze_excitation_layer(
                self.cnn_cell(query1, filters, size), out_dim=filters, ratio=4)
            answer1 += self.squeeze_excitation_layer(
                self.cnn_cell(answer1, filters, size), out_dim=filters, ratio=4)
            
            # shape = query1.shape.as_list()
            # print('query1', query1)
            query = tf.reshape(query1, [-1, shape[1], shape[2]*shape[3]])
            query = self.dense(query, 64, activation='relu')
            answer = tf.reshape(answer1, [-1, shape[1], shape[2]*shape[3]])
            answer = self.dense(answer, 64, activation='relu')
            print('result_q: ', query)
            print('result_a: ', answer)
            query_result.append(query)
            answer_result.append(answer)

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

    def dense(self, inputs, units, activation):
        if activation == 'relu':
            return tf.layers.dense(inputs, units, activation=tf.nn.relu,
                                   use_bias=True, kernel_initializer=self.initializer)
        if activation == 'leaky_relu':
            return tf.layers.dense(inputs, units, activation=tf.nn.leaky_relu,
                                   use_bias=True, kernel_initializer=self.initializer)
        if activation == 'selu':
            return tf.layers.dense(inputs, units, activation=tf.nn.selu,
                                   use_bias=True, kernel_initializer=self.initializer)
        return tf.layers.dense(inputs, units, activation=activation,
                                   use_bias=True, kernel_initializer=self.initializer)
    def batch_norm(self, input_x):
        return tf.layers.batch_normalization(input_x)

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
            print('self.sigma_q: ', self.sigma_q)
            print('self.sigma_a: ', self.sigma_a)
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