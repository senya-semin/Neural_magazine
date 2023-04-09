import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Brain():
    """
    Пока предсказнаия сугубо по прибыли
    """
    def __init__(self, memory = 500, discount = 0.95, epsilon_greedy=1.,
            epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-2) -> None:
        self.memory = memory
        self.gamma = discount
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        self.build_model()

    def build_model(self):
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(64, input_shape=(self.memory, 1), return_sequences=True))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.LSTM(32, return_sequences=False))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=self.lr), metrics=['accuracy'])

    def predict(self, timeline):
        timeline = np.array([timeline])
        return self.model.predict(timeline)
    
    def learn(self, history: list, actual_income: float, prediction: list):
        mean_income = np.mean(history)
        # На сколько текущий доход отличается от среднего
        if actual_income < 0 and mean_income < 0:
            income_change = mean_income/actual_income - 1
        elif mean_income < 0:
            income_change = 1
        else:
            income_change = actual_income/mean_income - 1
        prediction[np.argmax(prediction)] += income_change
        #Нормализуем данныеу
        if any(i <= 0 for i in prediction):
            prediction = [i + np.abs(np.min(prediction)) + 0.01 for i in prediction]
        prediction = [i/sum(prediction) for i in prediction]
        self.model.fit(np.array([history]), np.array([prediction]), epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay