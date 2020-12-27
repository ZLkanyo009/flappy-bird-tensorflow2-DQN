#!/usr/bin/env python
#============================ 导入所需的库 ===========================================
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense

import cv2
import sys
import random
import numpy as np
from collections import deque
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'

sys.path.append("game/")
import wrapped_flappy_bird as game

GAME = 'FlappyBird' # 游戏名称
ACTIONS = 2 # 2个动作数量
ACTIONS_NAME=['不动','起飞']  #动作名
GAMMA = 0.99 # 未来奖励的衰减
OBSERVE = 10000. # 训练前观察积累的轮数
EPSILON = 0.0001
REPLAY_MEMORY = 50000 # 观测存储器D的容量
BATCH = 32 # 训练batch大小
old_time = 0

class MyNet(Model):
    def __init__(self):
        super(MyNet, self).__init__()
        self.c1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a1_1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        #self.d1 = Dropout(0.2)  # dropout层

        self.c2_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a2_1 = Activation('relu')  # 激活层
        self.c2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a2_2 = Activation('relu')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层

        self.c3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a3_1 = Activation('relu')  # 激活层
        self.c3_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a3_2 = Activation('relu')  # 激活层
        self.c3_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a3_3 = Activation('relu')  # 激活层
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(ACTIONS, activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))

    def call(self, x):
        #print(x.shape)
        x = self.c1_1(x)
        #print(x.shape)
        x = self.a1_1(x)
        x = self.p1(x)

        x = self.c2_1(x)
        x = self.a2_1(x)
        x = self.c2_2(x)
        x = self.a2_2(x)
        x = self.p2(x)

        x = self.c3_1(x)
        x = self.a3_1(x)
        x = self.c3_2(x)
        x = self.a3_2(x)
        x = self.c3_3(x)
        x = self.a3_3(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        return y

def trainNetwork(istrain):
#============================ 模型创建与加载 ===========================================

    # 模型创建
    net1 = MyNet()
#============================ 配置模型 ===========================================
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6, epsilon=1e-08)  #1e-6

    epsilon = EPSILON
    t = 0 #初始化TIMESTEP

    # 加载保存的网络参数
    checkpoint_save_path = "./model/FlappyBird"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        net1.load_weights(checkpoint_save_path)
    else:
        print('-------------train new model-----------------')

#============================ 加载(搜集)数据集 ===========================================

    # 打开游戏
    game_state = game.GameState()

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    D = deque()

    #初始化状态并且预处理图片，把连续的四帧图像作为一个输入（State）
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print(s_t.shape)

    # 开始训练
    while True:
        # 根据输入的s_t,选择一个动作a_t
        readout_t = net1(tf.expand_dims(tf.constant(s_t, dtype=tf.float32), 0))
        print(readout_t)
        a_t_to_game = np.zeros([ACTIONS])
        action_index = 0

        #贪婪策略，有episilon的几率随机选择动作去探索，否则选取Q值最大的动作
        if random.random() <= epsilon and istrain:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t_to_game[action_index] = 1
        else:
            print("-----------net choice----------------")
            action_index = np.argmax(readout_t)
            print("-----------index----------------")
            print(action_index)
            a_t_to_game[action_index] = 1

        #执行这个动作并观察下一个状态以及reward
        x_t1_colored, r_t, terminal, score = game_state.frame_step(a_t_to_game)
        print("============== score ====================")
        print(score)

        rank_file_r = open("rank.txt","r")
        best = int(rank_file_r.readline())
        rank_file_r.close()
        #if score_one_round >= best:
        #    test = True
        best_checkpoint_save_path = "./best/FlappyBird"
        if score > best:
            net1.save_weights(best_checkpoint_save_path)
            rank_file_w = open("rank.txt","w")
            rank_file_w.write("%d" % score)
            print("********** best score updated!! *********")
            rank_file_w.close()
        if score >= best:
            f = open("scores.txt","a")
            f.write("========= %d ========== %d \n" % (t+old_time, score))
            f.close()

        a_t = np.argmax(a_t_to_game, axis=0)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        s_t_D = tf.convert_to_tensor(s_t, dtype=tf.float32)
        a_t_D = tf.constant(a_t, dtype=tf.int32)
        r_t_D = tf.constant(r_t, dtype=tf.float32)
        s_t1_D = tf.constant(s_t1, dtype=tf.float32)
        terminal = tf.constant(terminal, dtype=tf.float32)

        # 将观测值存入之前定义的观测存储器D中
        D.append((s_t_D, a_t_D, r_t_D, s_t1_D, terminal))
        #如果D满了就替换最早的观测
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 更新状态，不断迭代
        s_t = s_t1
        t += 1

#============================ 训练网络 ===========================================

        # 观测一定轮数后开始训练
        if (t > OBSERVE) and istrain:
            # 随机抽取minibatch个数据训练
            print("==================start train====================")
            
            minibatch = random.sample(D, BATCH)

            # 获得batch中的每一个变量
            b_s = [d[0] for d in minibatch]
            b_s = tf.stack(b_s, axis=0)

            b_a = [d[1] for d in minibatch]
            b_a = tf.expand_dims(b_a, axis=1)
            b_a = tf.stack(b_a, axis=0)

            b_r = [d[2] for d in minibatch]
            b_r = tf.stack(b_r, axis=0)

            b_s_ = [d[3] for d in minibatch]
            b_s_ = tf.stack(b_s_, axis=0)

            b_done = [d[4] for d in minibatch]
            b_done = tf.stack(b_done, axis=0)

            q_next = tf.reduce_max(net1(b_s_), axis=1)
            q_truth = b_r + GAMMA * q_next* (tf.ones(32) - b_done)

            # 训练
            with tf.GradientTape() as tape:
                q_output = net1(b_s)
                index = tf.expand_dims(tf.constant(np.arange(0, BATCH), dtype=tf.int32), 1)
                index_b_a = tf.concat((index, b_a), axis=1)
                q = tf.gather_nd(q_output, index_b_a)
                loss = tf.losses.MSE(q_truth, q)
                print("loss = %f" % loss)
                gradients = tape.gradient(loss, net1.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net1.trainable_variables))

            # 每1000轮保存一次网络参数
            if (t+old_time) % 1000 == 0:
                print("=================model save====================")
                net1.save_weights(checkpoint_save_path)

        # 打印信息

        print("TIMESTEP", (t+old_time), "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t, \
           "|  Q_MAX %e \n" % np.max(readout_t))
        # write info to files

def main():
    trainNetwork(istrain = False)

if __name__ == "__main__":
    main()
