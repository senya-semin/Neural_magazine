import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def choise(location):
    x = np.random.randint(0, location[0].size)
    y = np.random.randint(0, len(location))
    return (y, x)


def happines(location, adress, tolerance):
    y = adress[0]
    x = adress[1]
    income = []
    angry = 0
    if all(point > 0 for point in adress) and all(point < len(location) - 1 for point in adress):
        for i in (
            (location[y+1][x]), (location[y-1][x]),
            (location[y][x+1]), (location[y][x-1]),
            (location[y+1][x+1]), (location[y+1][x-1]),
            (location[y-1][x+1]), (location[y-1][x-1])
        ):
            income += [i]
    elif any(point == 0 for point in adress):
        if all(point == 0 for point in adress):
            for i in (
                (location[y][x+1]), (location[y-1][x]), (location[y-1][x+1])
            ):
                income += [i]
        elif y == 0 and x != 0 and x != len(location) - 1:
            for i in (
                (location[y+1][x]),
                (location[y][x+1]), (location[y][x-1]),
                (location[y+1][x+1]), (location[y+1][x-1]),
            ):
                income += [i]
        elif y != 0 and y != len(location) - 1 and x == 0:
            for i in (
                (location[y+1][x]), (location[y-1][x]),
                (location[y][x+1]),
                (location[y+1][x+1]),
                (location[y-1][x+1])
            ):
                income += [i]
        elif y == len(location) - 1 and x == 0:
            for i in (
                (location[y][x+1]), (location[y-1][x]),
                (location[y-1][x+1]),
            ):
                income += [i]
        elif y == 0 and x == len(location) - 1:
            for i in (
                (location[y+1][x]), (location[y][x-1]),
                (location[y+1][x-1]),
            ):
                income += [i]
    elif any(point == len(location) - 1 for point in adress):
        if all(point == len(location) - 1 for point in adress):
            for i in (
                (location[y-1][x]),
                (location[y][x-1]),
                (location[y-1][x-1])
            ):
                income += [i]
        elif y == len(location) - 1 and x != 0 and x != len(location) - 1:
            for i in (
                (location[y-1][x]),
                (location[y][x+1]), (location[y][x-1]),
                (location[y-1][x+1]), (location[y-1][x-1])
            ):
                income += [i]
        elif y != len(location) - 1 and y != 0 and x == len(location) - 1:
            for i in (
                (location[y+1][x]), (location[y-1][x]),
                (location[y][x-1]),
                (location[y+1][x-1]),
                (location[y-1][x-1])
            ):
                income += [i]
    income = np.mean(income)
    angry = income / location[y][x]
    if angry >= tolerance:
        return False
    else:
        return True


def shiller(location, intolerance):
    agent = choise(location)
    run = happines(location, agent, intolerance)
    if run:
        agent_income = location[agent[0]][agent[1]]
        if agent_income > 0.1 and agent_income < 1:
            houses = np.column_stack(np.where(location < agent_income))
            #print(houses)
            new_house = houses[np.random.choice(range(len(houses)))]
            renovation_agent = location[new_house[0]][new_house[1]]
            location[new_house[0]][new_house[1]] = agent_income
            location[agent[0]][agent[1]] = renovation_agent
    return location