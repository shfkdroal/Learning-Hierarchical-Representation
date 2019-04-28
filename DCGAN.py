
import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import ast
#import Extract_ExpertTrajactories
import graphic
import matplotlib.pyplot as plt
import math
from lib import *

#Game Param
w = 60
agentSpeed = 1
Num_product_category = 4
Num_product_for_each = 2
Num_desire = 2
Maximum_inventory = 10
Maximum_object_recognition = 45 #200 ->45
StateDim = 2 + (Maximum_inventory + 1) + Maximum_object_recognition
ActionDim = 4
State_and_Action_Dim = StateDim + ActionDim
Num_col = 2 * StateDim + 1
#Gloabal Variable 1
AgentCoord = [0, 0]
QualityPopupList = []

BatchSize = 8
NE = 10
UniformNoise = 0.3 #Epsilon
should_shuffle = True

currentState = np.zeros([BatchSize, StateDim, 1, 1], dtype=float)

class Agent:
    def __init__(self, x=0, y=0):
        self.xcor = x
        self.ycor = y
    def setx(self, x):
        self.xcor = x
    def sety(self, y):
        self.ycor = y

#class definition
class Product:
    def __init__(self, x=0, y=0, k=0, idx=0, qf=0):
        self.kind = k
        self.idx_within_kind = idx
        self.quality_factor = qf
        self.xCoord = x
        self.yCoord = y

    def getProductInfo(self):
        print("kind: {0}, idx: {1}, quality: {2}".format(self.kind, self.idx_within_kind, self.quality_factor))

    def getProducCoord(self):
        print("y: {0}, y: {1}".format(self.xCoord, self.yCoord))


class State:
    def __init__(self, x=[], internal=[], num_product=0, coord=[0, 0]):
        self.Num_Prod = num_product

        self.Num_have = 0

        self.current_coord = coord
        self.External_state_Buffer = x
        self.Internal_state_Buffer = internal

        for ii in range(Maximum_object_recognition):
            self.External_state_Buffer.append(0)

        for ii in range(Maximum_inventory+1):
            self.Internal_state_Buffer.append(0)

        self.Internal_state_Buffer[Maximum_inventory] = random.randint(1, Num_desire) # + 1

    def init(self, x=[], internal=[], num_product=0, coord=[0, 0]):
        self.Num_Prod = num_product

        self.Num_have = 0

        self.current_coord = coord
        self.External_state_Buffer = x
        self.Internal_state_Buffer = internal

        for ii in range(Maximum_object_recognition):
            self.External_state_Buffer.append(0)

        for ii in range(Maximum_inventory+1):
            self.Internal_state_Buffer.append(0)

        self.Internal_state_Buffer[Maximum_inventory] = random.randint(1, Num_desire) # + 1

    """
        self.kind = k
        self.idx_within_kind = idx
        self.quality_factor = qf

        self.xCoord = x
        self.yCoord = y

    """

    #State Initialization
    def Update_External_State_0(self, productList):
        self.current_coord = [0, 0]
        for np in range(len(productList)):
            v = productList[np]
            self.External_state_Buffer[5*np] = v.kind + 1
            self.External_state_Buffer[5*np+1] = v.idx_within_kind + 1
            self.External_state_Buffer[5*np+2] = 0
            self.External_state_Buffer[5*np+3] = v.xCoord + 1
            self.External_state_Buffer[5*np+4] = v.yCoord + 1
            self.Num_Prod += 1

    #Nurture Information when colide
    def Update_External_State_1(self, productList, colideIdx, batch_index):

        if colideIdx == -2:
            #the product was already handled
            for np in range(len(productList)):
                v = productList[np]
                self.External_state_Buffer[5 * np] = v.kind + 1
                self.External_state_Buffer[5 * np + 1] = v.idx_within_kind + 1
                self.External_state_Buffer[5 * np + 2] = 0
                self.External_state_Buffer[5 * np + 3] = v.xCoord + 1
                self.External_state_Buffer[5 * np + 4] = v.yCoord + 1

            self.External_state_Buffer[5 * (self.Num_Prod - 1)] = 0
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 1] = 0
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 2] = 0
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 3] = 0
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 4] = 0

            self.Num_Prod -= 1

        elif colideIdx != -1:
            #print(self.External_state_Buffer)
            self.External_state_Buffer[5 * colideIdx + 2] = productList[colideIdx].quality_factor + 1

        self.current_coord = AgentCoord
        self.record_state(batch_index)
        return

    def record_state(self, batch_index):

        global currentState

        """
                self.Num_Prod = num_product

                self.current_coord = coord
                self.External_state_Buffer = x
                self.Internal_state_Buffer = internal

                Plus 1 onto every elements

        """

        #f = open("transition.txt", 'a', encoding='utf-8')
        #f.write(str(self.current_coord + self.External_state_Buffer + self.Internal_state_Buffer) + '\n')
        currentState[batch_index, :, :, :] = np.array(self.current_coord + self.External_state_Buffer +
                                self.Internal_state_Buffer, dtype=float).reshape(1, StateDim, 1, 1)
        #f.close()
        return



def record_action(action_took):
    #print("action")
    #f = open("action.txt", 'a', encoding='utf-8')
    #f = open("transition.txt", 'a', encoding='utf-8')
    #f.write(str(action_took)+'\n')
    #f.close()
    return



def load_expert_trajectories():

    # prev State - action - nxt State

    f = open("transition.txt", "r", encoding='utf-8')

    prevState = np.array(ast.literal_eval(f.readline()))[0:StateDim] #Prev State
    line = f.readline()
    binnum = str(bin(round(float(line)/100)))[2:].zfill(4)
    newlist = []
    newlist[:0] = binnum
    action = np.array(newlist, dtype=float) #np.array([int(f.readline())], dtype=float)
    #nxtState = np.array(ast.literal_eval(f.readline()))
    Trajectories = np.concatenate((action, prevState), axis=0).reshape(1, State_and_Action_Dim)
    #Trajectories = np.concatenate((Trajectories, nxtState), axis=0).reshape(1, Num_col)

    for lines in f:
        if lines == '\n':
            break
        prevState = np.array(ast.literal_eval(lines))[0:StateDim]#nxtState
        line = f.readline()
        binnum = str(bin(round(float(line)/100)))[2:].zfill(4)
        newlist = []
        newlist[:0] = binnum
        action = np.array(newlist, dtype=float)
        """
        #print(action)
        if int(line) == 999:
            nxtState = np.full((2 + Maximum_inventory + 1 + Maximum_object_recognition,), np.NAN)
            temp = np.concatenate((action, prevState), axis=0)
            temp = np.concatenate((temp, nxtState), axis=0).reshape(1, Num_col)
            Trajectories = np.concatenate((Trajectories, temp), axis=0)
            nxtState = np.array(ast.literal_eval(f.readline()))
            #(Maximum_object_recognition,) #terminalState
        """
        #Trajectories = np.concatenate((action, prevState), axis=0).reshape(1, State_and_Action_Dim)
        temp = np.concatenate((action, prevState), axis=0).reshape(1, State_and_Action_Dim)
        Trajectories = np.concatenate((Trajectories, temp), axis=0)
        #Trajectories = np.concatenate((Trajectories, temp), axis=0)

    f.close
    #print(Trajectories)
    #print(Trajectories.shape)

    np.save('Trajectories', Trajectories)
    return Trajectories



def TrajectoryShuffle(tr):
    randrow = np.random.shuffle(tr)
    return


def viz_grid(Xs, padding):
    N, H, W, C = Xs.shape
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size + 1)
    grid_width = W * grid_size + padding * (grid_size + 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = padding, H + padding
    for y in range(grid_size):
        x0, x1 = padding, W + padding
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    return


def conv2d(input, kernel_size, stride, num_filter, name='conv2d'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        #print(input.get_shape())
        filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

        W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def conv2d_transpose(input, kernel_size, stride, num_filter, name='conv2d_transpose'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [kernel_size, kernel_size, num_filter, input.get_shape()[3]]
        output_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, num_filter])

        W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d_transpose(input, W, output_shape, stride_shape, padding='SAME') + b


def fc(input, num_output, name='fc'):
    with tf.variable_scope(name):
        num_input = input.get_shape()[1]
        W = tf.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [num_output], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, W) + b

def batch_norm(input, is_training):
    out = tf.contrib.layers.batch_norm(input, decay=0.99, center=True, scale=True,
                                       is_training=is_training, updates_collections=None)
    return out

def leaky_relu(input, alpha=0.2):
    return tf.maximum(alpha * input, input)


class DCGAN(object):
    def __init__(self):

        self.actionList = tf.placeholder(tf.float32, [6, 4, 1, 1])
        self.feedState_gen = tf.placeholder(tf.float32, [None, StateDim, 1, 1]) #lets try

        self.num_epoch = NE
        self.batch_size = BatchSize #=32
        self.indices = [0]*BatchSize
        self.log_step = 50
        self.visualize_step = 200
        self.code_size = 6432
        self.learning_rate = 1e-4
        self.vis_learning_rate = 1e-2
        self.recon_steps = 100
        self.actmax_steps = 100

        self._dis_called = False
        self._gen_called = False

        self.tracked_noise = np.random.normal(0, 1, [64, self.code_size])

        self.real_input = tf.placeholder(tf.float32, [None, State_and_Action_Dim, 1, 1])
        #32, 32, 3 -> State_and_Action_Dim
        self.real_label = tf.placeholder(tf.float32, [None, 1])
        self.fake_label = tf.placeholder(tf.float32, [None, 1])
        self.noise = tf.placeholder(tf.float32, [None, StateDim, 1, 1], name='noise')
        #[None, self.code_size] -> 32, 213, 1, 1

        self.is_train = tf.placeholder(tf.bool, name='is_training')

        self.recon_sample = tf.placeholder(tf.float32, [1, 32, 32, 3])
        self.actmax_label = tf.placeholder(tf.float32, [1, 1])

        with tf.variable_scope('actmax'):
            self.actmax_code = tf.get_variable('actmax_code', [1, self.code_size],
                                               initializer=tf.constant_initializer(0.0))

        self._init_ops()

    def _discriminator(self, input):
        # We have multiple instances of the discriminator in the same computation graph,
        # so set variable sharing if this is not the first invocation of this function.
        with tf.variable_scope('dis', reuse=self._dis_called):
            print(input.get_shape())
            self._dis_called = True
            dis_conv1 = conv2d(input, 4, 1, 16, 'conv1')
            dis_lrelu1 = leaky_relu(dis_conv1)
            dis_conv2 = conv2d(dis_lrelu1, 4, 1, 32, 'conv2')
            dis_batchnorm2 = batch_norm(dis_conv2, self.is_train)
            dis_lrelu2 = leaky_relu(dis_batchnorm2)
            #dis_conv3 = conv2d(dis_lrelu2, 4, 1, 64, 'conv3')
            #dis_batchnorm3 = batch_norm(dis_conv3, self.is_train)
            #dis_lrelu3 = leaky_relu(dis_batchnorm3)
            print(dis_lrelu2.get_shape())
            dis_reshape3 = tf.reshape(dis_lrelu2, [-1, 62 * 1 * 32]) #4 -> 28
            dis_fc4 = fc(dis_reshape3, 1, 'fc4')
            return dis_fc4

    def _generator(self, input):
        with tf.variable_scope('gen', reuse=self._gen_called):
            self._gen_called = True
            gen_fc1 = fc(tf.reshape(input, [-1, StateDim]), 4 * 4 * 128, 'fc1')
            gen_reshape1 = tf.reshape(gen_fc1, [-1, 64, 2, 2])
            gen_batchnorm1 = batch_norm(gen_reshape1, self.is_train)
            gen_lrelu1 = leaky_relu(gen_batchnorm1)
            gen_fc1_1 = tf.reshape(gen_reshape1, [-1, 16, 2, 2])
            gen_reshape1_1 = tf.reshape(gen_fc1_1, [-1, 16 * 2 * 2])
            gen_batchnorm1_1 = batch_norm(gen_reshape1_1, self.is_train)
            gen_lrelu1_1 = leaky_relu(gen_batchnorm1_1)
            #gen_conv2 = conv2d_transpose(gen_lrelu1, 4, 2, 64, 'conv2')
            #gen_batchnorm2 = batch_norm(gen_conv2, self.is_train)
            #gen_lrelu2 = leaky_relu(gen_batchnorm2)

            #gen_conv3 = conv2d_transpose(gen_lrelu1, 4, 2, 64, 'conv3') #32 -> 64
            #gen_batchnorm3 = batch_norm(gen_conv3, self.is_train)
            #gen_lrelu3 = leaky_relu(gen_batchnorm3)

            #gen_conv4 = conv2d_transpose(gen_lrelu1, 4, 2, 32, 'conv4') #3->32
            #gen_batchnorm4 = batch_norm(gen_conv4, self.is_train) #added by Yoon
            #gen_lrelu4 = leaky_relu(gen_batchnorm4)
            gen_reshape2 = tf.reshape(gen_lrelu1_1, [-1, 32 * 64])
            gen_fc2 = fc(gen_reshape2, 4, 'fc1_1')
            gen_fc2_reshaped = tf.reshape(gen_fc2, [-1, 4, 1, 1])
            print(gen_fc2_reshaped.get_shape()) #batchsize=32, 4, 1, 1

            GenOutput = tf.concat([gen_fc2_reshaped, self.feedState_gen], axis=1, name="GenOut")

            self.StateUpdateBatch_for_genBuffer(gen_fc2_reshaped)
            #gen_sigmoid4 = tf.sigmoid(gen_conv4)
            #add a fc, remove some convs
            #should combine action and state within this generator

            return GenOutput #gen_sigmoid4

    #v == action result numpy array from policy net
    def StateUpdateBatch_for_genBuffer(self, v):
        for u in range(self.batch_size):
            dist_2 = tf.reduce_sum((self.actionList - tf.matrix_transpose(v[u, :, :, :])) ** 2, reduction_indices=1)
            self.indices[u] = tf.argmin(dist_2, 0)

        self._generator_nxt_state(self.indices)

    def _generator_nxt_state(self, action_indices):

        for k in range(self.batch_size):
            if np.random.rand(1) <= UniformNoise:
                action_indices[k] = np.random.randint(0, 5, 1, dtype=int)
                rndX = np.random.randint(0, 10, 1, dtype=int)[0]
                rndY = np.random.randint(0, 10, 1, dtype=int)[0]
                agent.ycor = rndX
                agent.xcor = rndY
            # nextState = currentState
            if action_indices[k] == 0:
                move_left(k)
            elif action_indices[k] == 1:
                move_right(k)
            elif action_indices[k] == 2:
                move_up(k)
            elif action_indices[k] == 3:
                move_down(k)
            elif action_indices[k] == 4:
                get(k)
            elif action_indices[k] == 5:
                cook(k)
            # return nextState


    def _loss(self, labels, logits):
        #print(labels.get_shape(), logits.get_shape())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def _reconstruction_loss(self, generated, target):
        loss = tf.nn.l2_loss(generated - target)
        return tf.reduce_mean(loss)

    # Define operations
    def _init_ops(self):

        self.fake_samples_op = self._generator(self.noise)
        fake_through_dis = self._discriminator(self.fake_samples_op)
        # self.dis_loss_op = None
        self.dis_loss_op = self._loss(self.real_label, self._discriminator(self.real_input)) + \
                           self._loss(self.fake_label, fake_through_dis)
        # self.gen_loss_op = None
        self.gen_loss_op = self._loss(self.real_label, fake_through_dis)

        ################################################################################
        # Prob 2-1: fix the definition of these operations                             #
        ################################################################################

        # dis_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # self.dis_train_op = dis_optimizer.minimize(self.dis_loss_op)
        dis_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           "dis")
        # print(dis_train_vars)
        self.dis_train_op = optimizer.minimize(self.dis_loss_op, var_list=dis_train_vars)

        # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        gen_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           "gen")
        # print(gen_train_vars)
        # self.gen_train_op = gen_optimizer.minimize(self.gen_loss_op)
        self.gen_train_op = optimizer.minimize(self.gen_loss_op, var_list=gen_train_vars)


    # Training function
    def train(self, sess, train_samples):
        sess.run(tf.global_variables_initializer())
        num_train = train_samples.shape[0]
        step = 0

        generator_state_buffer = train_samples[:, ActionDim:State_and_Action_Dim, :, :][:]

        action_List = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        action_List = action_List.reshape(6, 4, 1, 1)

        # smooth the loss curve so that it does not fluctuate too much
        smooth_factor = 0.95
        plot_dis_s = 0
        plot_gen_s = 0
        plot_ws = 0

        temp_batch = 1

        dis_losses = []
        gen_losses = []
        max_steps = int(self.num_epoch * (num_train // self.batch_size))
        print('Start training ...')
        for epoch in range(self.num_epoch):
            for i in range(num_train // self.batch_size):
                step += 1

                batch_samples = train_samples[i * self.batch_size: (i + 1) * self.batch_size]
                #noise = np.random.normal(0, 1, [self.batch_size, self.code_size])

                noise = generator_state_buffer[0:self.batch_size, :, :, :]
                noise + np.random.normal(0.0, 0.5, noise.shape)
                #print("noise size: ", noise.shape)
                #self.fake_samples_op = self._generator(self.noise)
                #currentState = np.zeros(32*213).reshape(self.batch_size, 213, 1, 1) #Give one batch (Episodes)
                #fake_action_and_state = sess.run([self.fake_samples_op], feed_dict=one_feed_gen)


                zeros = np.zeros([self.batch_size, 1])
                ones = np.ones([self.batch_size, 1])

                ################################################################################
                # Prob 2-1: complete the feed dictionary                                       #
                ################################################################################

                #self.noise = noise
                #self.fake_label = zeros
                #self.real_label = ones
                #self.real_input = batch_samples
                #dis_feed_dict = {}

                dis_feed_dict = {self.noise: noise, self.fake_label: zeros, self.real_label: ones,
                                 self.real_input: batch_samples,
                                 self.is_train: True, self.feedState_gen: currentState, self.actionList: action_List}

                ################################################################################
                #                               END OF YOUR CODE                               #
                ################################################################################

                _, dis_loss = sess.run([self.dis_train_op, self.dis_loss_op], feed_dict=dis_feed_dict)


                # gen_feed_dict = {}
                gen_feed_dict = {self.noise: noise, self.real_label: ones,
                                 self.is_train: True, self.actionList: action_List, self.feedState_gen: currentState}

                if should_shuffle:
                    np.random.shuffle(generator_state_buffer)
                generator_state_buffer[0:self.batch_size, :, :, :] = currentState
                if should_shuffle:
                    np.random.shuffle(train_samples)


                _, gen_loss = sess.run([self.gen_train_op, self.gen_loss_op], feed_dict=gen_feed_dict)

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                dis_losses.append(plot_dis_s / plot_ws)
                gen_losses.append(plot_gen_s / plot_ws)

                if step % self.log_step == 0:
                    print('Iteration {0}/{1}: dis loss = {2:.4f}, gen loss = {3:.4f}'.format(step, max_steps,
                                                                                             dis_loss, gen_loss))

            """
            ax1 = plt.subplot(111)
            ax1.imshow(viz_grid(self.generate(self.tracked_noise), 1))
            plt.show()
            plt.plot(dis_losses)
            plt.title('discriminator loss')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.show()

            plt.plot(gen_losses)
            plt.title('generator loss')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.show()
            """
        print('... Done!')

    # Find the reconstruction of one input sample
    def reconstruct_one_sample(self, sample):
        actmax_init_val = tf.convert_to_tensor(np.zeros(self.code_size).reshape(1, self.code_size),
                                                   dtype=tf.float32)
        sess.run(self.actmax_code.assign(actmax_init_val))  # assigned (no need to be in feed dict)
        last_reconstruction = None
        last_loss = None
        for i in range(self.recon_steps):
            ################################################################################
            # Prob 2-4: complete the feed dictionary                                       #
            # skip this part when working on problem 2-1 and come back for problem 2-4     #
            ################################################################################

            # recon_feed_dict = {}
            recon_feed_dict = {
                self.is_train: False,
                self.real_input: sample
            }

            ################################################################################
            #                               END OF YOUR CODE                               #
            ################################################################################

            run_ops = [self.recon_loss_op, self.reconstruct_op, self.actmax_sample_op]
            last_loss, _, last_reconstruction = sess.run(run_ops, feed_dict=recon_feed_dict)
        return last_loss, last_reconstruction

        # Find the reconstruction of a batch of samples
    def reconstruct(self, samples):
        reconstructions = np.zeros(samples.shape)
        total_loss = 0
        for i in range(samples.shape[0]):
            loss, reconstructions[i:i + 1] = self.reconstruct_one_sample(samples[i:i + 1])
            total_loss += loss
        return total_loss / samples.shape[0], reconstructions

        # Generates a single sample from input code
    def generate_one_sample(self, code):
        gen_vis_feed_dict = {self.noise: code, self.is_train: False, }

        generated = sess.run(self.fake_samples_op, feed_dict=gen_vis_feed_dict)
        return generated

        # Generates samples from input batch of codes
    def generate(self, codes):
            # [1, self.code_size] (1,64)
        generated = np.zeros((codes.shape[0], 32, 32, 3))
        for i in range(codes.shape[0]):
            generated[i:i + 1] = self.generate_one_sample(codes[i:i + 1])
        return generated

        # Perform activation maximization on one initial code
    def actmax_one_sample(self, initial_code):
        actmax_init_val = tf.convert_to_tensor(initial_code, dtype=tf.float32)
        sess.run(self.actmax_code.assign(actmax_init_val))
        for i in range(self.actmax_steps):
            actmax_feed_dict = {
                self.actmax_label: np.ones([1, 1]),
                self.is_train: False
            }
            _, last_actmax = sess.run([self.actmax_op, self.actmax_sample_op], feed_dict=actmax_feed_dict)
        return last_actmax  # return code sized, single vector (feed forwrd) - the training session is done within actmax

        # Perform activation maximization on a batch of different initial codes
    def actmax(self, initial_codes):
            # initial_codes == np.random.random([64, dcgan.code_size])
        actmax_results = np.zeros((initial_codes.shape[0], 32, 32, 3))
        for i in range(initial_codes.shape[0]):
            actmax_results[i:i + 1] = self.actmax_one_sample(initial_codes[i:i + 1])
        return actmax_results.clip(0, 1)




########Simulator#########

AgentCoord = [0, 0]

state = State()# Agent State
ProductList = []
ProdCoordList = []
CoordList = [0]*121

success_prob = 0.9

agent = Agent(0, 0)

#util function (MISC)
def xy_rand_generator():
    x = random.randrange(0, 10, 1)
    y = random.randrange(0, 10, 1)
    coord = [x, y]
    if coord not in ProdCoordList:
        ProdCoordList.append(coord)
        return coord
    else:
        coord = xy_rand_generator()
        return coord

def spread_product(num_product, num_kind, batch_index):

    for k in range(num_kind):
        for p in range(num_product):
            coords = xy_rand_generator()
            single_product = Product(coords[0], coords[1], k, p, random.randint(2, 10))
            # coord, kind, index, quality
            ProductList.append(single_product)

    k = 0
    for v in ProductList:
        random.seed(v.kind)
        CoordList[(11*v.xCoord) + v.yCoord] = k
        k += 1

    state.Update_External_State_0(ProductList)
    collisionTest(batch_index)


def collisionTest(batch_index):
    global AgentCoord
    colideIdx = -1
    X = agent.xcor
    Y = agent.ycor
    AgentCoord = [X, Y]
    k =CoordList[11*AgentCoord[0] + AgentCoord[1]]
    if k != -1:
        colideIdx = k
        #if colideIdx >= len(ProdCoordList):
        newK = 0
        for cd in ProdCoordList:
            if cd[0] == X and cd[1] == Y:
                colideIdx = newK
                CoordList[11 * AgentCoord[0] + AgentCoord[1]] = newK
            newK += 1

    state.Update_External_State_1(productList=ProductList, colideIdx=colideIdx, batch_index=batch_index)
    return


#action definition
def move_left_(batch_index):
    x = agent.xcor
    if x > 0:
        x -= agentSpeed
    agent.setx(x)
    collisionTest(batch_index)
def move_right_(batch_index):
    x = agent.xcor
    if x < 10:
        x += agentSpeed
    agent.setx(x)
    collisionTest(batch_index)
def move_up_(batch_index):
    y = agent.ycor
    if y < 10:
        y += agentSpeed
    agent.sety(y)
    collisionTest(batch_index)
def move_down_(batch_index):
    y = agent.ycor
    if y > 0:
        y -= agentSpeed
    agent.sety(y)
    collisionTest(batch_index)

def move_left(batch_index):
    #wn.delay(20)
    record_action(100)
    t = random.uniform(0, 10)
    if t > 1:
        x = agent.xcor
        if x > 0:
            x -= agentSpeed
        agent.setx(x)
    elif t <= 0.2:
        move_right_(batch_index)
        return
    elif t <= 0.6:
        move_up_(batch_index)
        return
    elif t <= 1:
        move_down_(batch_index)
        return
    collisionTest(batch_index)

def move_right(batch_index):
    #wn.delay(20)
    record_action(200)
    t = random.uniform(0, 10)
    if t > 1:
        x = agent.xcor
        if x < 10:
            x += agentSpeed
        agent.setx(x)
    elif t <= 0.2:
        move_left_(batch_index)
        return
    elif t <= 0.6:
        move_up_(batch_index)
        return
    elif t <= 1:
        move_down_(batch_index)
        return
    collisionTest(batch_index)

def move_up(batch_index):
    #wn.delay(20)
    record_action(300)
    t = random.uniform(0, 10)
    if t > 1:
        y = agent.ycor
        if y < 10:
            y += agentSpeed
        agent.sety(y)
    elif t <= 0.2:
        move_down_(batch_index)
        return
    elif t <= 0.6:
        move_right_(batch_index)
        return
    elif t <= 1:
        move_left_(batch_index)
        return
    collisionTest(batch_index)

def move_down(batch_index):
    record_action(400)
    t = random.uniform(0, 10)
    if t > 1:
        y = agent.ycor
        if y > 0:
            y -= agentSpeed
        agent.sety(y)
    elif t <= 0.2:
        move_up_(batch_index)
        return
    elif t <= 0.6:
        move_right_(batch_index)
        return
    elif t <= 1:
        move_left_(batch_index)
        return
    collisionTest(batch_index)

def cook(batch_index):
    record_action(999)
    global state
    state.Internal_state_Buffer = [0] * (Maximum_inventory + 1)
    state.Internal_state_Buffer[Maximum_inventory] = random.randint(0, Num_desire - 1)
    state.Num_have = 0

    state.init(x=[], internal=[], num_product=0, coord=[0, 0])

    global ProductList
    global ProdCoordList
    ProdCoordList = []
    ProductList = []
    global CoordList
    CoordList = [0] * 121
    random.seed(os.times())
    spread_product(Num_product_for_each, Num_product_category, batch_index)

    return


def get(batch_index):
    record_action(500)
    global ProductList
    global ProdCoordList
    global CoordList
    global state
    global AgentCoord

    k = CoordList[11*AgentCoord[0] + AgentCoord[1]]

    if k == -1:
        return

    scaled121 = 11*ProdCoordList[k][0] + ProdCoordList[k][1]
    CoordList[scaled121] = -1

    state.Internal_state_Buffer[state.Num_have] = ProductList[k].kind
    if state.Num_have < Maximum_inventory - 1:
        state.Num_have += 1

    del ProductList[k]
    del ProdCoordList[k]

    state.Update_External_State_1(productList=ProductList, colideIdx=-2, batch_index=batch_index)


#load_expert_trajectories()
TrajectExpert = np.load('Trajectories.npy')
# #Trajectory buffer ASS' 2132 * 427, Batch size = 30

T_shape = TrajectExpert.shape
rsize = T_shape[0]
csize = T_shape[1]

if should_shuffle:
    TrajectoryShuffle(TrajectExpert)

spread_product(Num_product_for_each, Num_product_category, 0) #(num_product, num_kind)

TrajectExpert = TrajectExpert.reshape(rsize, csize, 1, 1)

tf.reset_default_graph()
set_seed(21)

train_samples = TrajectExpert


with tf.Session() as sess:
    with tf.device('/cpu:0'):
        dcgan = DCGAN()
        sess.run(tf.global_variables_initializer())
        dcgan.train(sess, train_samples)
        dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dis')
        gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gen')
        saver = tf.train.Saver(dis_var_list + gen_var_list)
        saver.save(sess, 'model/dcgan')




