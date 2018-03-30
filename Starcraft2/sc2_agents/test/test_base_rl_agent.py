import unittest
from sc2_agents.base_rl_agent import DQNCNN, BaseRLAgent

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from pysc2.lib import actions
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]
_SELECT_POINT = actions.FUNCTIONS.select_point.id

class TestDQN(unittest.TestCase):
  # def __init__(self):
  #   pass

  def setUp(self):
    self.dqn = DQNCNN()
    self.agent = BaseRLAgent()

  def test_DQN(self):
    x = torch.rand(5, 17, 84, 84)
    x = Variable(torch.Tensor(x))

    output = self.dqn(x)
    self.assertEqual(5, output.size()[0])
    self.assertEqual(2, output.size()[1])
    self.assertEqual(84, output.size()[2])
    self.assertEqual(84, output.size()[3])

  def test_get_random_action(self):
    s = np.random.rand(17, 84, 84)
    action = self.agent.get_action(s)
    # self.assertTrue(action[0] in [0, 1])
    # self.assertTrue(action[1:].shape[0] == 2)
    self.assertTrue(action < 2*84*84)

  def test_get_greedy_action(self):
    self.agent._epsilon.increment(100)
    s = np.random.rand(17, 84, 84)
    action = self.agent.get_action(s)
    # print(action)
    # self.assertTrue(action[0] in [0, 1])
    # self.assertTrue(action[1:].shape[0] == 2)
    self.assertTrue(action < 2 * 84 * 84)

  def test_gather(self):
    s = np.array([[1, 2], [3, 4]])
    # print(s.shape)
    s = torch.from_numpy(s)
    # print("s", s.size())

    index = torch.from_numpy(np.array([[1], [0]])).long()
    # print("index", index.size())
    out = s.gather(1, index)
    # print(out)


    s = np.array([[
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [9, 10, 11, 12]
      ],
      [
        [11, 12, 13, 14],
        [14, 15, 16, 17],
        [19, 20, 21, 22]
      ]
    ])
    # print("new one =============")
    s = torch.from_numpy(s)
    # print("s", s.size())

    index = torch.from_numpy(np.array([[
        [1],
        [1],
        [1],
      ],
      [
        [2],
        [2],
        [2]
      ]
    ])).long()
    # print("index", index.size())
    out = s.gather(2, index)
    # print(out)

  def test_some_other_gather(self):
    # some data
    orig = torch.rand(2, 3, 5)
    # print(orig)
    # where we want to put the data
    # notice: idx.size() is equal orig.size()
    # idx will be dimension zero index of the target))
    idx = np.random.randint(0, 15, [2, 1])
    # print("idx", idx, idx.shape)
    idx = torch.from_numpy(idx)
    # idx = torch.LongTensor([[2], [2], [2]])
    # notice: t1.size(1) is equal orig.size(1)
    # t1 = torch.zeros(3, 5).scatter_(0, idx, orig)
    # print(t1)

    # zeros of the right size
    t2 = torch.rand(2, 3, 5)
    # print("t2", t2)
    t2 = t2.view(2, -1)
    # print("t2", t2)

    copy = t2.gather(1, idx)
    # print("copy", copy)


if __name__ == "__main__":
  unittest.main()


