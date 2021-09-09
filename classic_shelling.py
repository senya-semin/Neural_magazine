import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def choise(location):
    x = np.random.randint(0, location[0].size)
    y = np.random.randint(0, len(location))
    return (y,x)

def happines(location, adress, intolerance):
    y = adress[0]
    x = adress[1]
    races = []
    angry = 0
    if all(point > 0 for point in adress) and all(point < len(location) - 1 for point in adress):
        for i in (
            (location[y+1][x]), (location[y-1][x]),
            (location[y][x+1]), (location[y][x-1]),
            (location[y+1][x+1]), (location[y+1][x-1]),
            (location[y-1][x+1]), (location[y-1][x-1])
            ):
            races += [i]
    elif any(point == 0 for point in adress):
        if all(point == 0 for point in adress):
            for i in (
                (location[y][x+1]), (location[y-1][x]), (location[y-1][x+1])
                ):
                races += [i]
        elif y == 0 and x != 0 and x != len(location) - 1:
            for i in (
                (location[y+1][x]),
                (location[y][x+1]), (location[y][x-1]),
                (location[y+1][x+1]), (location[y+1][x-1]),
                ):
                races += [i]
        elif y != 0 and y != len(location) - 1 and x == 0:
            for i in (
                (location[y+1][x]), (location[y-1][x]),
                (location[y][x+1]),
                (location[y+1][x+1]),
                (location[y-1][x+1])                
                ):
                races += [i]
    elif any(point == len(location) - 1 for point in adress):
        if all(point == len(location) - 1 for point in adress):
            for i in (
                (location[y-1][x]),
                (location[y][x-1]),
                (location[y-1][x-1])
                ):
                races += [i]
        elif y == len(location) - 1 and x != 0 and x != len(location) - 1:
            for i in (
                (location[y-1][x]),
                (location[y][x+1]), (location[y][x-1]),
                (location[y-1][x+1]), (location[y-1][x-1])
                ):
                races += [i]
        elif y != len(location) - 1 and y != 0 and x == len(location) - 1:
            for i in (
                (location[y+1][x]), (location[y-1][x]),
                (location[y][x-1]),
                (location[y+1][x-1]),
                (location[y-1][x-1])
                ):
                races += [i]
    for race in races:
        if race != location[adress] and race != 0:
            angry += race
    if round(angry/8, 2) <= intolerance:
        return False
    else: 
        return True

def shiller(location, intolerance):
    agent = choise(location)
    run = happines(location, agent, intolerance)
    if run:
        houses = np.column_stack(np.where(location == 0))
        new_house = houses[np.random.choice(range(len(houses)))]
        location[new_house[0]][new_house[1]] = location[agent[0]][agent[1]]
        location[agent[0]][agent[1]] = 0
    return location

def update(frame):
    global city
    global intolerance
    city = shiller(city, intolerance)
    matrice.set_array(city)


intolerance = 0.4

empty = 0
white = 1
black = 2

p_empty = 0.1
p_white = 0.6
p_black = 0.3

steps = 200

city = np.random.choice([empty, white, black], (100,100), p = [p_empty, p_white, p_black])


fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
matrice = ax.matshow(city)
plt.colorbar(matrice)

plt.imshow(city, interpolation= "none")
plt.show()

for i in range(int(1e5)):
    city = shiller(city, intolerance)
plt.imshow(city, interpolation= "none")
plt.show()

ani = animation.FuncAnimation(fig, update, frames = 15000, interval = 1)
plt.show()


