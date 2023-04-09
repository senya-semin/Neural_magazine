import numpy as np

class Agent():
    t = 0

    def __init__(self, y, x, income= 50000000):
        self.price = {}
        self.knowledge = {}
        self.magazine = 0
        self.y = y
        self.x = x
        self.lenght = []
        self.income = income
        self.e = 1.1 

    def InitPrice(self, magazines, starting_price):
        for magazine in range(magazines):
            self.price[f"{magazine}"] = [starting_price, 0]
        self.knowledge = self.price

    def choose(self, coordinates):
        magazine_x = coordinates[:,1]
        magazine_y = coordinates[:,0]
        self.lenght = [np.sqrt((magazine_x[i] - self.x)**2 + (magazine_y - self.y)**2) for i in range(len(magazine_x))]
        for i in range(len(self.lenght)):
            if self.lenght[i] == 0:
                self.lenght[i] = 1
        self.magazine = np.argmin([self.price[f"{i}"] * self.lenght[i] for i in range(len(self.price))])

    def choose_integral(self, magazines):
        choose = []
        choose_ = []
        self.lenght = [np.sqrt((spot.x - self.x)**2 + (spot.y - self.y)**2) for spot in magazines]
        for i in range(len(self.lenght)):
            if self.lenght[i] == 0:
                self.lenght[i] = 1
        self.magazine = [self.e ** (self.price[f"{i}"][0] / self.income) * self.lenght[i] for i in range(len(self.price))]
        for i in range(len(self.lenght)):
            if self.price[f"{i}"][0] <= self.income:
                choose += [self.magazine[i]]
                choose_ += [i]
        if len(choose) >= 1:
            self.magazine = choose_[np.argmin(choose)]
        else:
            self.magazine = np.random.randint(0, len(self.price))

    def GoToShopping(self, actual_price):
        self.price[f"{self.magazine}"] = [actual_price, Agent.t]
        if actual_price >= self.income:
            self.magazine = None

    def CollectInformation(self, dict):
        for i in range(len(dict)):
            if dict[f"{i}"][0] < self.knowledge[f"{i}"][0] and dict[f"{i}"][1] > self.knowledge[f"{i}"][1]:
                self.knowledge[f"{i}"] = dict[f"{i}"]
    
    def UpdateInformation(self):
        self.price = self.knowledge.copy()

    def UpdateTime(self):
        Agent.t += 0.01

    def DropTime(self):
        Agent.t = 0


