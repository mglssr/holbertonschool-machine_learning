#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

ppl = ["Farrah", "Fred", "Felicia"]
fruits = ["apples", "bananas", "oranges", "peaches"]
colors = ["r", "yellow", "#ff8000", "#ffe5b4"]
data = len(fruit)
ystack = np.zeros(len(ppl))
for i in range(data):
    plt.bar(ppl, fruit[i], bottom=ystack, color=colors[i], width=0.5)
    ystack = ystack + fruit[i]
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.ylim([0, 80])
plt.legend(fruits, loc=1, prop={'size': 9})
plt.show()
