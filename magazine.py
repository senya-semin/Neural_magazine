from Brain import *
import time

class Magazine():
    def __init__(self, y:int, x:int, init_price:float, rent: float, inflation = 0.2, name = None, neural = False, memory = 500, allow_close = False, day_until_close = 70) -> None:
        self.x = x
        self.y = y
        self.price = init_price
        self.clients = 0
        self.income = 0
        self.rent = rent
        self.open_days = 0
        self.neural = neural
        self.capital = 0
        self.inflation = inflation
        self.status = True
        self.day_until_close = day_until_close
        self.day_close = 0
        self.allow_close = allow_close
        if name is not None:
            self.name = name
        if neural:
            self.memory = memory
            self.brain = Brain(self.memory)
            self.strategy = [0., 0., 0.]
            

    def update_price_random(self):
        self.price += np.random.choice([-10, 0, 10])

    def update_price_neural(self, history):
        if self.open_days < self.memory:
            self.update_price_random()
            return 
        self.brain.update_epsilon()
        self.strategy = self.brain.predict(history)[0]
        if np.random.rand() <= self.brain.epsilon:
            self.update_price_random()
            return
        self.price += [-10, 0, 10][np.argmax(self.strategy)]

    def learn(self, history):
        time.sleep(0.5)
        self.brain.learn(history, self.income, self.strategy)

    def colletct_income(self):
        if self.status:
            self.income = self.price * self.clients - self.rent
            self.capital = self.capital * (1- self.inflation) + self.income
            if self.day_close >= self.day_until_close and self.allow_close:
                print("Я закрылся")
                self.status = False
                self.clients = 0
                self.income = 0
                self.capital = 0
                self.price = 0
            self.day_close += 1
            if self.capital > 0:
                self.day_close = 0

    def collect_history(self, history):
        if self.brain.politic == 'income':
            return [history.income[f"{self.name}"][-self.memory :], history.prices[f"{self.name}"][-self.memory :]]
        elif self.brain.politic == 'capital':
            return [history.capital[f"{self.name}"][-self.memory :], history.prices[f"{self.name}"][-self.memory :]]
        elif self.brain.politic == 'profitability':
            return [history.income[f"{self.name}"][-self.memory :], history.clients[f"{self.name}"][-self.memory :], history.prices[f"{self.name}"][-self.memory :]]

    def new_day(self, history):
        if self.status:
            if self.neural and (self.open_days > self.memory):
                self.learn(history)
            self.open_days += 1
            self.clients = 0