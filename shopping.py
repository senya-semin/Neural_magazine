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

size = 25
agent = Agent(1,1)
city = np.empty((size,size), dtype=object)
iterations = 100

for i in range(size):
    for j in range(size):
        city[i][j] = Agent(i,j)

magazine_0 = Magazine(np.random.randint(0,25), np.random.randint(0,25), 0)
magazine_1 = Magazine(np.random.randint(0,25), np.random.randint(0,25), 1)
magazine_2 = Magazine(np.random.randint(0,25), np.random.randint(0,25), 2)
magazine_3 = Magazine(np.random.randint(0,25), np.random.randint(0,25), 3)
magazine_4 = Magazine(np.random.randint(0,25), np.random.randint(0,25), 4)

magazines = [magazine_0,magazine_1,magazine_2,magazine_3,magazine_4]
locations = collect_coordinates(magazines)
print(locations)

for i in range(size):
    for j in range(size):
        city[i][j].InitPrice(5, 100)
        city[i][j].choose(locations)

price = {}
income = {}
clients = {}
clients_ = {}
for magazine in magazines:
    price[f"{magazine.name}"] = []
    income[f"{magazine.name}"] = []
    clients_[f"{magazine.name}"] = []

for _ in range(iterations):
    for magazine in magazines:
        clients[f"{magazine.name}"] = 0
        magazine.choose_action(1)

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
            clients[f"{city[i][j].magazine}"] += 1

    for magazine in magazines:
        magazine.CollectClients(clients[f"{magazine.name}"])
        price[f"{magazine.name}"] += [magazine.price]
        income[f"{magazine.name}"] +=[ magazine.income]
        clients_[f"{magazine.name}"] += [magazine.clients]

    city[0][0].UpdateTime()

fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
for magazine in clients_.values():
    sns.lineplot(x=range(len(magazine)), y = magazine)
plt.show()
for magazine in price.values():
    sns.lineplot(x=range(len(magazine)), y = magazine)
plt.show()
for magazine in income.values():
    sns.lineplot(x=range(len(magazine)), y = magazine)
plt.show()
