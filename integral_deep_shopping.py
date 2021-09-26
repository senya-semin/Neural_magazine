import re
import numpy as np
from agent import Agent
from magazine import Magazine
from m import neighbors
import seaborn as sns
import matplotlib.pyplot as plt
from integral_shelling import shiller, cubic
#from deep_shopping import GetItDone


def collect_coordinates(magazines):
    return np.array([[magazine.y, magazine.x] for magazine in magazines])


def collect_knowledge(neighbors):
    return [neighbor.price for neighbor in neighbors]


def collect_concurents(magazine, magazines,  train=True):
    concurents = []
    for concurent in magazines:
        if int(concurent) != magazine:
            if train:
                concurents += [magazines[concurent][-2]]
            else:
                concurents += [magazines[concurent][-1]]
    return np.array(concurents)


def field(magazine, starategys):
    strategy_today = []
    for starategy in range(len(starategys)):
        if starategy != magazine:
            strategy_today += [starategys[f"{starategy}"][1]]
    return np.sum(strategy_today, axis=0)


def education(Magazine, price, clients, city, concurents):
    my_price = price[f"{Magazine.name}"]
    state = np.array(clients[-Magazine.learn_size - 1:-1])
    action = np.where(np.round(Magazine.strategy, 2) ==
                      round((Magazine.price - my_price[-2]), 2))
    reward = Magazine.income / (city.size)
    next_state = np.array(clients[-Magazine.learn_size:])
    current_concurents = concurents[-2]
    next_concurents = concurents[-1]
    transition = [state, action, reward, next_state,
                  current_concurents, next_concurents]
    Magazine.Remember(transition)
    Magazine.Learn(transition)
    Magazine.UpdateEpsilon()


def GetItDone(city, magazines, locations, iteration, clients, cl, income, concurents, price, clients_, train=False, plotting=True, init_price=100, integral=False):
    size = len(city)
    for i in range(size):
        for j in range(size):
            city[i][j].InitPrice(len(magazines), init_price)

    for magazine in magazines:
        magazine.price = city[magazine.y][magazine.x].income
        print(magazine.price)

    for magazine in magazines:
        cl[f"{magazine.name}"] = 0

    for i in range(size):
        for j in range(size):
            city[i][j].choose(locations)
            city[i][j].GoToShopping(magazines[city[i][j].magazine].price)
            if city[i][j].magazine:
                cl[f"{city[i][j].magazine}"] += 1

    for magazine in magazines:
        magazine.CollectClients(cl[f"{magazine.name}"])

    cl = {}

    for day in range(iteration):
        for magazine in magazines:
            city[magazine.y][magazine.x].price[f"{magazine.name}"] = [
                magazine.price, city[0][0].t]

        for magazine in magazines:
            cl[f"{magazine.name}"] = 0
            magazine.choose_action(
                income[f"{magazine.name}"], concurents[f"{magazine.name}"][-1], train=train)

        for i in range(size):
            for j in range(size):
                neighbor = neighbors(city, i, j, 1)
                knowledges = collect_knowledge(neighbor)
                for knowledge in knowledges:
                    city[i][j].CollectInformation(knowledge)

        for i in range(size):
            for j in range(size):
                city[i][j].UpdateInformation()
                city[i][j].choose_integral(locations)
                city[i][j].GoToShopping(magazines[city[i][j].magazine].price)              
                if city[i][j].magazine is not None:
                    cl[f"{city[i][j].magazine}"] += 1

        for magazine in magazines:
            magazine.CollectClients(cl[f"{magazine.name}"])
            price[f"{magazine.name}"] += [magazine.price]
            income[f"{magazine.name}"] += [magazine.income]
            clients[f"{magazine.name}"] += [magazine.clients]
            education(
               magazine, price, income[f"{magazine.name}"], city, concurents[f"{magazine.name}"])

        for magazine in magazines:
            #print(len(income[f"{magazine.name}"][-magazine.learn_size:]))
            concurents[f"{magazine.name}"] += [magazine.Predict(
                income[f"{magazine.name}"][-magazine.learn_size:], concurents[f"{magazine.name}"][-1])]
            concurents[f"{magazine.name}"] = concurents[f"{magazine.name}"][-2:]
        for magazine in magazines:
            concurents[f"{magazine.name}"][1] = field(
                magazine.name, concurents)
        print(day)

        city[0][0].UpdateTime()

    city[0][0].DropTime()

    if plotting:
        for magazine in clients_.values():
            sns.lineplot(x=range(len(magazine))[
                         :iteration], y=magazine[-iteration:])
        plt.show()
        for magazine in price.values():
            sns.lineplot(x=range(len(magazine))[
                         :iteration], y=magazine[-iteration:])
        plt.show()
        for magazine in income.values():
            sns.lineplot(x=range(len(magazine))[
                         :iteration:], y=magazine[-iteration:])
        plt.show()

##

##


size = 50
iterations = 500
incomes = [x**3 for x in np.arange(0.1, 1, 0.001)]
distribution = np.random.choice(incomes, (size, size))
tolerance = 0.6
plt.imshow(distribution, interpolation="none")
plt.show()
for _ in range(int(1e6)):
    distribution = shiller(distribution, tolerance)
city = np.empty((size, size), dtype=object)

for i in range(size):
    for j in range(size):
        city[i][j] = Agent(i, j, distribution[i][j] * 1300 + 200)

magazines = [Magazine(np.random.randint(0, size),np.random.randint(0, size), i) for i in range(5)]
locations = np.array([[magazine.y, magazine.x] for magazine in magazines])
print(locations)

plt.imshow(distribution, interpolation="none")
plt.show()

for location in locations:
    distribution[location[0]][location[1]] = 10000000

plt.imshow(distribution, interpolation="none")
plt.show()

step = {}
for magazine in magazines:
    step[f"{magazine.name}"] = 0
for i in range(size):
    for j in range(size):
        city[i][j].InitPrice(len(magazines), 150)
        city[i][j].choose(locations)
        step[f"{city[i][j].magazine}"] += 1

for magazine in magazines:
    magazine.CollectClients(step[f"{magazine.name}"])

price = {}
income = {}
clients = {}
clients_ = {}
concurents = {}


for magazine in magazines:
    price[f"{magazine.name}"] = [100 for _ in range(550)]
    income[f"{magazine.name}"] = [magazine.income for _ in range(550)]
    clients_[f"{magazine.name}"] = [magazine.clients for _ in range(550)]
    concurents[f"{magazine.name}"] = [magazine.strategy for _ in range(2)]
    magazine.build_model()

GetItDone(city, magazines, locations, 25000, clients_, clients, income, concurents,
          price, clients_, train=False, plotting=True, init_price=150, integral=True)
