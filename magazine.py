from Brain import *

class Magazine():
    def __init__(self, y:int, x:int, init_price:float, rent: float, name = None, neural = False, memory = 500) -> None:
        self.x = x
        self.y = y
        self.price = init_price
        self.clients = 0
        self.income = 0
        self.rent = rent
        if name is not None:
            self.name = name
        if neural:
            self.memory = memory
            self.brain = Brain(self.memory)
            self.prediction = [0., 0., 0.]
            

    def update_price_random(self):
        strategy = np.random.choice(["up","hold","down"])
        if strategy == "up":
            self.price += 10
        elif strategy == "down":
            self.price -= 10

    def update_price_neural(self, history):
        self.brain.update_epsilon()
        self.prediction = self.brain.predict(history)[0]
        if np.random.rand() <= self.brain.epsilon:
            return self.update_price_random()
        self.price += [-10, 0, 10][np.argmax(self.prediction)]

    def learn(self, history):
        self.brain.learn(history, self.income, self.prediction)

    def colletct_income(self):
        self.income = self.price * self.clients - self.rent

    def new_day(self):
        self.clients = 0