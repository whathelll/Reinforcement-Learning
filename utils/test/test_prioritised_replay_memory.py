import unittest
from utils.replay_memory import PrioritisedReplayMemory, Transition


class TestPrioritisedReplayMemory(unittest.TestCase):
  def setUp(self):
    self.memory = PrioritisedReplayMemory(capacity=10, e=0.1, alpha=0.5)

  def test_append(self):
    for i in range(20):
      a = Transition([0, 1, 2, 3], 0, [4, 5, 6, 7], 0, True)
      self.memory.push(a)
    self.assertEqual(len(self.memory.memory), 10)

  # haven't tested the proportional code yet
  def test_sample(self):
    for i in range(10):
      a = Transition([0, 1, 2, i], 0, [4, 5, 6, i*i], 0, True)
      self.memory.push(a)

    [s, a, s1, r, done], indices = self.memory.sample(2)
    self.assertEqual(s.shape, (2, 4))
    self.assertEqual(a.shape, (2, 1))
    self.assertEqual(s1.shape, (2, 4))
    self.assertEqual(r.shape, (2, 1))
    self.assertEqual(done.shape, (2, 1))
    # print(self.memory.sample())

  def test_update(self):
    for i in range(10):
      a = Transition([0, 1, 2, i], 0, [4, 5, 6, i*i], 0, True)
      self.memory.push(a)

    self.memory.update([1, 3], [2, 5])
    self.assertEqual(self.memory.errors[1], 2.1)
    self.assertEqual(self.memory.errors[3], 5.1)
    # print(self.memory)


if __name__ == "__main__":
  unittest.main()