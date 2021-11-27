
#%%
from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# x = []
# y = []
# fig, ax = plt.subplots()
# def animate(i):
#     pt = randint(1,9)
#     x.append(i)
#     y.append(pt)

#     ax.clear()
#     ax.plot(x, y)
#     ax.set_xlim([0,20])
#     ax.set_ylim([0,10])
# ani = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)
# plt.show()
#%%
plt.ion()
for i in range(100):
    x = range(i)
    y = range(i)
    # plt.gca().cla() # optionally clear axes
    plt.plot(x, y)
    plt.title(str(i))
    plt.draw()
    plt.pause(0.1)

plt.show(block=True)

