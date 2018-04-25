import numpy as np
import matplotlib.pyplot as plt

class RewardHistory(list):
  def plot(self):
    R = np.array(self)
    mu = np.mean(R, axis=0)
    std = np.std(R, axis=0)
    f, axarr = plt.subplots(1, 2)
    f.set_figheight(5)
    f.set_figwidth(20)
    f.subplots_adjust(hspace=0.2)
    axarr[0].plot(mu)
    axarr[0].set_title('Mean reward')
    axarr[0].set_xlabel("Episodes")
    axarr[0].set_ylabel("Reward")
    axarr[1].set_title('Std')
    axarr[1].set_xlabel("Episodes")
    axarr[1].set_ylabel("std")
    axarr[1].plot(std)
    plt.show()

