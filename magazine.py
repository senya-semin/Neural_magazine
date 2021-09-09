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
        self.strategy = np.arange(-1, 1 + 0.1, 0.1)
        self.income = 0
        self.clients = 0
        self.y = y
        self.x = x
        self.name = name
        self.memory = deque(maxlen=memory_size)
        self.max_memory = memory_size
        self.gamma = discount
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.income = 0
        self.learn_size = len(self.strategy)
        self.cut = 0
        self.pref_rew = 0
        self.predict = tf.convert_to_tensor(
            np.reshape(np.random.rand(1, 21), (1, self.learn_size)))
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
            90, activation = "tanh")(self.reshape_2)
        self.price_pred = tf.keras.layers.Dense(
            21,)(self.dot_up)
        self.model = tf.keras.Model(
            inputs=[self.input_price, self.input_clients, self.input_concurents],
            outputs=self.price_pred
        )
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(lr=self.lr)
        )

        # Иной вариант - хуже, но можно дорабатывать:
        # self.input_price = tf.keras.Input(
        #     shape=(self.strategy.size, ))
        # self.input_clients = tf.keras.Input(
        #     shape=(self.learn_size, 1))
        # self.input_concurents = tf.keras.Input(
        #     shape=(self.strategy.size, ))
        # self.dense = tf.keras.layers.Dense(
        #     20)(self.input_price)
        # self.concurents = tf.keras.layers.Multiply()([self.dense, self.input_concurents])
        # self.reshape = tf.keras.layers.Reshape((1, 20))(self.concurents)
        # # self.body_price = tf.keras.layers.Dense(
        # #     40)(self.reshape)
        # self.body_clients = tf.keras.layers.LSTM(
        #     20, input_shape=(self.learn_size, 1))(self.input_clients)
        # self.reshape_ = tf.keras.layers.Reshape((20, 1))(self.body_clients)
        # self.dot = tf.keras.layers.Dot(
        #     axes=(1,2), normalize=True)([self.reshape, self.reshape_])
        # self.reshape_1 = tf.keras.layers.Reshape((400, 1))(self.dot)
        # self.reshape_2 = tf.keras.layers.Reshape((1, 400))(self.reshape_1)
        # self.dot_up = tf.keras.layers.Dense(
        #     80, activation = "sigmoid")(self.reshape_2)
        # self.price_pred = tf.keras.layers.Dense(
        #     20, activation = "tanh")(self.dot_up)
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
        state = np.array(state[-self.strategy.size:])
        current_step = tf.convert_to_tensor(
            np.reshape(state, (1, self.learn_size)))
        concurents = tf.convert_to_tensor(
            np.reshape(concurents, (1, self.strategy.size)))
        return self.model.predict(
            [self.next_predict, current_step, concurents])[0][0]

    def choose_action(self, state, concurents, train=False):
        self.cut = 0
        if self.price - np.max(self.strategy) <= 0:
            self.cut = int(np.abs(self.price - np.max(self.strategy)) / 0.1)
        if train:
            self.price = self.price + \
                np.random.choice(self.strategy[self.cut:])
            return self.price
        if np.random.rand() <= self.epsilon:
            self.price = self.price + \
                np.random.choice(self.strategy[self.cut:])
            return self.price
        state = np.array(state[-self.strategy.size:])
        current_step = tf.convert_to_tensor(
            np.reshape(state, (1, self.learn_size)))
        concurents = tf.convert_to_tensor(
            np.reshape(concurents, (1, self.strategy.size)))
        self.q_value = np.argmax(self.model.predict(
            [self.next_predict, current_step, concurents])[0][0][self.cut:])
        self.price = self.price + self.strategy[self.q_value + self.cut]

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
        self.next_predict = tf.convert_to_tensor(
            np.reshape(self.model.predict([self.next_predict, current_step, next_concurents])[0][0][self.cut:], (1, self.strategy.size)))
        target = (reward + self.gamma *
                  (self.model.predict([self.next_predict, next_step, next_concurents])[0][0][action]) - self.pref_rew)
        target_all = self.model.predict(
            [self.predict, current_step, current_concurents])
        target_all[0][0][action] = target
        if self.cut > 0:
            for cut in range(self.cut):
                target_all[0][0][cut] = - np.abs(target) 
        self.pref_rew = reward
        self.model.fit([self.predict, current_step, current_concurents], target_all, epochs=1, verbose=0)

    def UpdateEpsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def CollectClients(self, clients):
        self.clients = clients
        self.income = self.clients * self.price - 7500 - self.price

# m = Magazine(1,1,1)
# print(m.model.summary())
# m.ShowModel()