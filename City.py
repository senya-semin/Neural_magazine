import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

from Magazine import *
from History import *
from Agent import *
from integral_shelling import shiller


class City:
    def __init__(
        self,
        size: int,
        distribution,
        tolerance: float,
        magazine_namber,
        max_income=1500,
        min_income=200,
        neural=False,
        memory=500,
        allow_concurents = False,
        allow_close = False
    ) -> None:
        self.size = size
        self.magazines = []
        self.history = History()
        self.neural = neural
        self.memory = memory
        self.allow_concurents = allow_concurents
        self.allow_close = allow_close
        self.initialize_cityzen(
            distribution, tolerance, max_income - min_income, min_income
        )
        self.initialize_magazine(magazine_namber)

    def initialize_cityzen(self, distribution, tolerance, max_income, min_income):
        incomes = [distribution(x) for x in np.arange(0.1, 1, 0.001)]
        people = np.random.choice(incomes, (self.size, self.size))

        for _ in range(int(1e6)):
            distribution = shiller(people, tolerance)
        self.cityzen = np.empty((self.size, self.size), dtype=object)

        for i in range(self.size):
            for j in range(self.size):
                self.cityzen[i][j] = Agent(
                    i, j, distribution[i][j] * max_income + min_income
                )

    def initialize_magazine(self, magazine_namber):
        self.magazines = []
        for i in range(magazine_namber):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            self.magazines += [
                Magazine(
                    y=y,
                    x=x,
                    init_price=self.cityzen[y][x].income * 0.8,
                    rent=self.cityzen[y][x].income * 9,
                    name=i,
                    neural=self.neural,
                    memory=self.memory,
                    allow_close= self.allow_close
                )
            ]

    def prepare_city(self):
        knowledge = {}
        for magazine in self.magazines:
            knowledge[f"{magazine.name}"] = [magazine.price, 0]
            self.history.clients[f"{magazine.name}"] = [magazine.clients]
            self.history.prices[f"{magazine.name}"] = [magazine.price]
            self.history.income[f"{magazine.name}"] = [magazine.income]
            self.history.capital[f"{magazine.name}"] = [magazine.capital]

        for i in range(self.size):
            for j in range(self.size):
                self.cityzen[i][j].price = knowledge
                self.cityzen[i][j].knowledge = knowledge

    def iterate(self, days, preparation=True):
        if preparation:
            self.prepare_city()

        for day in range(days):
            for magazine in self.magazines:
                if magazine.status:
                    self.cityzen[magazine.y][magazine.x].price[f"{magazine.name}"][
                        0
                    ] = magazine.price
                    if self.neural:
                        magazine.update_price_neural(
                            [self.history.income[f"{magazine.name}"][-self.memory :], self.history.prices[f"{magazine.name}"][-self.memory :]]
                        )
                    else:
                        magazine.update_price_random()

            for i in range(self.size):
                for j in range(self.size):
                    for magazine in self.magazines:
                        if magazine.status:
                            continue
                        self.cityzen[i][j].knowledge.pop(str(magazine.name), None)
                        self.cityzen[i][j].price.pop(str(magazine.name), None)

            for i in range(self.size):
                for j in range(self.size):
                    neighbors = self.neighbors(self.cityzen, i, j, 1)
                    knowledges = [neighbor.price for neighbor in neighbors]
                    for knowledge in knowledges:
                        self.cityzen[i][j].CollectInformation(knowledge)

            for i in range(self.size):
                for j in range(self.size):
                    self.cityzen[i][j].UpdateInformation()
                    self.cityzen[i][j].choose_integral(self.magazines)
                    self.cityzen[i][j].GoToShopping(
                        self.magazines[self.cityzen[i][j].magazine].price
                    )
                    if self.cityzen[i][j].magazine is not None:
                        for magazine in self.magazines:
                            if magazine.name == self.cityzen[i][j].magazine:
                                magazine.clients += 1

            contentment = 0
            for magazine in self.magazines:
                magazine.colletct_income()
                self.history.clients[f"{magazine.name}"] += [magazine.clients]
                self.history.prices[f"{magazine.name}"] += [magazine.price]
                self.history.income[f"{magazine.name}"] += [magazine.income]
                self.history.capital[f"{magazine.name}"] += [magazine.capital]
                self.history.status[f"{magazine.name}"] = magazine.status
                contentment += magazine.clients
                magazine.new_day(
                    [self.history.income[f"{magazine.name}"][-self.memory :], self.history.prices[f"{magazine.name}"][-self.memory :]]
                )
            self.history.contentment += [contentment/(self.size**2)]

            self.cityzen[0][0].UpdateTime()

            if self.allow_concurents:
                knowledge = self.new_magazine(day, knowledge)
            print(day)

    def generate_location(self, day):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            for magazine in self.magazines:
                if magazine.status:
                    if x == magazine.x and y == magazine.y:
                        return self.generate_location(day)
            new_name = int(list(self.history.income.keys())[-1]) + 1
            self.magazines += [
                Magazine(
                    y=y,
                    x=x,
                    init_price=self.cityzen[y][x].income * 0.8,
                    rent=self.cityzen[y][x].income * 9,
                    name=new_name,
                    neural=self.neural,
                    memory=self.memory,
                    allow_close= self.allow_close
                )
            ]
            self.history.clients[f"{new_name}"] = np.zeros(day).tolist()
            self.history.prices[f"{new_name}"] = np.zeros(day).tolist()
            self.history.income[f"{new_name}"] = np.zeros(day).tolist()
            self.history.capital[f"{new_name}"] = np.zeros(day).tolist()

    def new_magazine(self, day, knowledge):
        if np.random.rand() < (1 - self.history.contentment[-1]) ** 2:
            print("Я родился!")
            self.generate_location(day + 2)
            knowledge[f"{self.magazines[-1].name}"] = [self.magazines[-1].price, self.cityzen[0][0].t]
            for i in range(self.size):
                for j in range(self.size):
                    self.cityzen[i][j].price[f"{self.magazines[-1].name}"] = [self.magazines[-1].price, self.cityzen[0][0].t]
                    self.cityzen[i][j].knowledge[f"{self.magazines[-1].name}"] = [self.magazines[-1].price, self.cityzen[0][0].t]
        return knowledge

    def neighbors(self, arr, i, j, d):
        """Return d-th neighbors of cell (i, j)"""

        def sliding_window(arr, window_size):
            """Construct a sliding window view of the array"""
            arr = np.asarray(arr)
            window_size = int(window_size)
            if arr.ndim != 2:
                raise ValueError("need 2-D input")
            if not (window_size > 0):
                raise ValueError("need a positive window size")
            shape = (
                arr.shape[0] - window_size + 1,
                arr.shape[1] - window_size + 1,
                window_size,
                window_size,
            )
            if shape[0] <= 0:
                shape = (1, shape[1], arr.shape[0], shape[3])
            if shape[1] <= 0:
                shape = (shape[0], 1, shape[2], arr.shape[1])
            strides = (
                arr.shape[1] * arr.itemsize,
                arr.itemsize,
                arr.shape[1] * arr.itemsize,
                arr.itemsize,
            )
            return as_strided(arr, shape=shape, strides=strides)

        w = sliding_window(arr, 2 * d + 1)

        ix = np.clip(i - d, 0, w.shape[0] - 1)
        jx = np.clip(j - d, 0, w.shape[1] - 1)

        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)

        quartal = w[ix, jx][i0:i1, j0:j1].ravel()
        for house in range(len(quartal)):
            if quartal[house] == arr[i][j]:
                quartal = np.delete(quartal, house)
                break
        return quartal

    def plot_magazines(self):
        magazine_distribution = np.zeros((self.size, self.size))
        for magazine in self.magazines:
            if magazine.status:
                magazine_distribution[magazine.y][magazine.x] = 1
        plt.imshow(magazine_distribution, interpolation="none")
        plt.show()

    def plot_cityzen(self):
        income_distribution = np.empty((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                income_distribution[i][j] = self.cityzen[i][j].income

        plt.imshow(income_distribution, interpolation="none")
        plt.show()
