import re
import numpy as np
from agent import Agent
from magazine import Magazine
from m import neighbors
import seaborn as sns
import matplotlib.pyplot as plt


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
    # print(state)
    action = np.where(np.round(Magazine.strategy, 2) ==
                      round((Magazine.price - my_price[-2]), 2))
    reward = Magazine.income / (city.size)
    next_state = np.array(clients[-Magazine.learn_size:])
    current_concurents = concurents[-2]
    next_concurents = concurents[-1]
    transition = [state, action, reward, next_state,
                  current_concurents, next_concurents]
    #print("!!!!!!!!!!!!!!!!!!", reward, "!!!!!!!!!!!!!!!!!!!!!!!!!")
    Magazine.Remember(transition)
    Magazine.Learn(transition)
    Magazine.UpdateEpsilon()


def GetItDone(city, magazines, locations, iteration, clients, cl, income, concurents, price, clients_, train=False, plotting=True):
    size = len(city)
    for i in range(size):
        for j in range(size):
            city[i][j].InitPrice(len(magazines), 100)

    for magazine in magazines:
        magazine.price = 100

    for magazine in magazines:
        cl[f"{magazine.name}"] = 0

    for i in range(size):
        for j in range(size):
            city[i][j].GoToShopping(magazines[city[i][j].magazine].price)
            cl[f"{city[i][j].magazine}"] += 1

    for magazine in magazines:
        magazine.CollectClients(cl[f"{magazine.name}"])

    cl = {}

    for day in range(iteration):
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
                city[i][j].choose(locations)
                city[i][j].GoToShopping(magazines[city[i][j].magazine].price)
                cl[f"{city[i][j].magazine}"] += 1

        for magazine in magazines:
            magazine.CollectClients(cl[f"{magazine.name}"])
            price[f"{magazine.name}"] += [magazine.price]
            income[f"{magazine.name}"] += [magazine.income]
            clients[f"{magazine.name}"] += [magazine.clients]
            education(
                magazine, price, income[f"{magazine.name}"], city, concurents[f"{magazine.name}"])

        for magazine in magazines:
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
###

###


size = 100
iterations = 500
city = np.empty((size, size), dtype=object)

for i in range(size):
    for j in range(size):
        city[i][j] = Agent(i, j)

magazine_0 = Magazine(np.random.randint(0, size), np.random.randint(0, size), 0)
magazine_1 = Magazine(np.random.randint(0, size), np.random.randint(0, size), 1)
magazine_2 = Magazine(np.random.randint(0, size), np.random.randint(0, size), 2)
magazine_3 = Magazine(np.random.randint(0, size), np.random.randint(0, size), 3)
magazine_4 = Magazine(np.random.randint(0, size), np.random.randint(0, size), 4)

magazines = [magazine_0, magazine_1, magazine_2, magazine_3, magazine_4]
locations = collect_coordinates(magazines)
print(locations)

step = {}
for magazine in magazines:
    step[f"{magazine.name}"] = 0
for i in range(size):
    for j in range(size):
        city[i][j].InitPrice(len(magazines), 100)
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

GetItDone(city, magazines, locations, 7500, clients_, clients, income,
          concurents, price, clients_, train=False, plotting=True)
