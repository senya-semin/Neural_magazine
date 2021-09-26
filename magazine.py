from typing import Optional
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
import pydot
import pydotplus
import graphviz

class Magazine():
    Transition = namedtuple(
        "Transition", ("state", "action", "reward", "next_state"))

    def __init__(self, y, x, name,
                 discount=0.95, epsilon_greedy=1.,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=1e-2, memory_size=500):
        self.price = 100
        # self.strategy = np.arange(self.price - 10, self.price + 10, 0.1)
        #self.strategy = np.arange(0.75, 1.25, 0.001)
        self.strategy = np.arange(-5, 5 + 0.1, 0.5)
        self.income = 0
        self.clients = 0
        self.y = y
        self.x = x
        self.name = name
        self.rent = 20000
        self.memory = deque(maxlen=memory_size)
        self.max_memory = memory_size
        self.gamma = discount
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.income = 0
        self.learn_size = 80 #len(self.strategy)
        self.cut = 0
        self.pref_rew = 0
        self.predict = tf.convert_to_tensor(
            np.reshape(np.random.rand(1, 21), (1, self.strategy.size)))
        self.next_predict = self.predict
        self.select_strategy = tf.convert_to_tensor(
            np.reshape(self.strategy, (1, self.strategy.size)))

        self.build_model()

    def build_model(self):

        
        self.input_price = tf.keras.Input(
            shape=(self.strategy.size, ))
        self.input_clients = tf.keras.Input(
            shape=(self.learn_size, 1))
        self.input_concurents = tf.keras.Input(
            shape=(self.strategy.size, ))
        self.dense = tf.keras.layers.Dense(
            21, activation = "tanh")(self.input_price)
        self.reshape_price = tf.keras.layers.Reshape((self.strategy.size, 1))(self.dense)
        self.reshape_concurents = tf.keras.layers.Reshape((1, self.strategy.size))(self.input_concurents)
        self.concurents = tf.keras.layers.Dot(axes = (2,1), normalize=False)([self.reshape_price, self.reshape_concurents])
        self.reshape = tf.keras.layers.Reshape((1, 441))(self.concurents)
        self.body_price = tf.keras.layers.Dense(
            40)(self.reshape)
        self.body_clients = tf.keras.layers.LSTM(
            40, input_shape=(self.learn_size, 1))(self.input_clients)
        self.reshape_ = tf.keras.layers.Reshape((40, 1))(self.body_clients)
        self.dot = tf.keras.layers.Dot(
            axes=(1,2), normalize=True)([self.body_price, self.reshape_])
        self.reshape_1 = tf.keras.layers.Reshape((1600, 1))(self.dot)
        self.reshape_2 = tf.keras.layers.Reshape((1, 1600))(self.reshape_1)
        self.dot_up = tf.keras.layers.Dense(
            200, activation = "softsign")(self.reshape_2)
        #self.dot_up = tf.keras.layers.Dense(100, activation = "tanh")(self.dot_up_)
        self.price_pred = tf.keras.layers.Dense(
            21)(self.dot_up)
        self.prediscount = tf.keras.layers.Dense(7, activation = "tanh")(self.price_pred)
        self.discount = tf.keras.layers.Dense(1, activation = "sigmoid")(self.prediscount)
        self.model = tf.keras.Model(
            inputs=[self.input_price, self.input_clients, self.input_concurents],
            outputs= [self.price_pred, self.discount]
        )
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(lr=self.lr)
        )

        # self.input_price = tf.keras.Input(
        #     shape=(self.strategy.size, ))
        # self.input_clients = tf.keras.Input(
        #     shape=(self.learn_size, 1))
        # self.input_concurents = tf.keras.Input(
        #     shape=(self.strategy.size, ))
        # self.dense = tf.keras.layers.Dense(
        #     21)(self.input_price)
        # self.reshape_price = tf.keras.layers.Reshape((self.strategy.size, 1))(self.dense)
        # self.reshape_concurents = tf.keras.layers.Reshape((1, self.strategy.size))(self.input_concurents)
        # self.concurents = tf.keras.layers.Dot(axes = (2,1), normalize=False)([self.reshape_price, self.reshape_concurents])
        # self.body_price_ = tf.keras.layers.Dense(11, activation = "softsign")(self.concurents)
        # self.body_price = tf.keras.layers.Dense(1, activation = "tanh")(self.body_price_)
        # self.price_reshape = tf.keras.layers.Reshape((self.strategy.size, 1))(self.body_price)
        # self.relu = tf.keras.layers.Dense(1, activation = "sigmoid")(self.price_reshape)
        # self.body_clients = tf.keras.layers.LSTM(
        #     21, input_shape=(self.learn_size, 1))(self.input_clients)
        # self.reshape_ = tf.keras.layers.Reshape((1, 21))(self.body_clients)
        # self.dot = tf.keras.layers.Dot(
        #     axes=(2, 1), normalize=True)([self.relu, self.reshape_])
        # self.dot_up_ = tf.keras.layers.Dense(12, activation = "tanh")(self.dot)
        # self.m = tf.keras.layers.Dense(1)(self.dot_up_)
        # self.dot_up = tf.keras.layers.Reshape((1,21))(self.m)
        # self.price_pred_ = tf.keras.layers.Dense(
        #     21, activation = "softsign")(self.dot_up)
        # self.price_pred = tf.keras.layers.Dense(21)(self.price_pred_)
        # self.model = tf.keras.Model(
        #     inputs=[self.input_price, self.input_clients, self.input_concurents],
        #     outputs=self.price_pred
        # )
        # self.model.compile(
        #     loss="mse",
        #     optimizer=tf.keras.optimizers.Adam(lr=self.lr)
        # )

    def ShowModel(self):
        return tf.keras.utils.plot_model(self.model, "model.png", show_shapes=True)

    def Remember(self, transition):
        self.memory.append(transition)

    def Predict(self, state, concurents,):
        state = np.array(state[-self.learn_size:])
        current_step = tf.convert_to_tensor(
            np.reshape(state, (1, self.learn_size)))
        concurents = tf.convert_to_tensor(
            np.reshape(concurents, (1, self.strategy.size)))
        return self.model.predict(
            [self.next_predict, current_step, concurents])[0][0][0]

    def choose_action(self, state, concurents, train=False):
        self.cut = 0
        if self.price - np.max(self.strategy) <= 0:
            self.cut = int(np.abs(self.price - np.max(self.strategy)) / (self.strategy[-1] - self.strategy[-2]))
        if train:
            self.price = self.price + \
                np.random.choice(self.strategy[self.cut:])
            return self.price
        if np.random.rand() <= self.epsilon:
            self.price = self.price + \
                np.random.choice(self.strategy[self.cut:])
            return self.price
        state = np.array(state[-self.learn_size:])
        current_step = tf.convert_to_tensor(
            np.reshape(state, (1, self.learn_size)))
        concurents = tf.convert_to_tensor(
            np.reshape(concurents, (1, self.strategy.size)))
        self.q_value = np.argmax(self.model.predict(
            [self.next_predict, current_step, concurents])[0][0][0][self.cut:])
        self.price = self.price + self.strategy[self.q_value + self.cut]
        #print(self.model.predict([self.next_predict, current_step, concurents]))

    def Learn(self, batch_sample):
        state, action, reward, next_state, current_concurents, next_concurents = batch_sample
        current_step = tf.convert_to_tensor(
            np.reshape(state, (1, self.learn_size)))
        next_step = tf.convert_to_tensor(
            np.reshape(next_state, (1, self.learn_size)))
        next_concurents = tf.convert_to_tensor(
            np.reshape(next_concurents, (1, self.strategy.size)))
        current_concurents = tf.convert_to_tensor(
            np.reshape(current_concurents, (1, self.strategy.size)))
        self.predict = self.next_predict
        #print(self.model.predict([self.next_predict, current_step, next_concurents])[0][0][0])
        self.next_predict = tf.convert_to_tensor(
            np.reshape(self.model.predict([self.next_predict, current_step, next_concurents])[0][0][0], (1, self.strategy.size)))
        self.gamma = self.model.predict([self.next_predict, next_step, next_concurents])[1][0][0]
        target = (reward + self.gamma *
                  (self.model.predict([self.next_predict, next_step, next_concurents])[0][0][0][action]) - self.pref_rew) 
        # if self.income <= 0:
        #     print("Имя магазина:", self.name)
        #     print("цена:", self.price)
        #     print("действие:", self.strategy[action])
        #     print("награда:",reward)
        #     print("таргет:", target)
        #     print("дисконт:", self.gamma)
        #     print(self.model.predict([self.next_predict, next_step, next_concurents]))
        # print(target)
        # print(reward)
        # print(self.model.predict([self.next_predict, next_step, next_concurents]))
        # if target >= 1:
        #     target = 1
        # elif target <= -1:
        #     target = -1
        target_all = self.model.predict(
            [self.predict, current_step, current_concurents])
        target_all[0][0][0][action] = target
        if self.income > 0:
            target_all[1][0][0] = self.gamma + 0.01
        else:
            target_all[1][0][0] = self.gamma - 0.1
        if self.cut > 0:
            for cut in range(self.cut):
                target_all[0][0][0][cut] = - np.abs(target) 
        self.pref_rew = reward
        self.model.fit([self.predict, current_step, current_concurents], target_all, epochs=1, verbose=0)

    def UpdateEpsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def warehouse_overflow(self, value):
        return  value ** self.price

    def CollectClients(self, clients):
        self.clients = clients
        self.warehose = 0
        if self.income <= 0:
            self.warehose = self.warehouse_overflow(1.007)
        self.income = self.clients * self.price - self.rent - self.price - self.warehose

# m = Magazine(1,1,1)
# print(m.model.summary())
# m.ShowModel()