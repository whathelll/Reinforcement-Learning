import unittest
from utils.replay_memory import ReplayMemory, Transition


class TestReplayMemory(unittest.TestCase):
  def setUp(self):
    self.memory = ReplayMemory(capacity=10)

  def test_append(self):
    for i in range(20):
      a = Transition([0, 1, 2, 3], 0, [4, 5, 6, 7], 0, True)
      self.memory.push(a)
    self.assertEqual(len(self.memory.memory), 10)

  def test_sample(self):
    for i in range(10):
      a = Transition([0, 1, 2, i], 0, [4, 5, 6, i*i], 0, True)
      self.memory.push(a)

    s, a, s1, r, done = self.memory.sample(2)
    self.assertEqual(s.shape, (2, 4))
    self.assertEqual(a.shape, (2, 1))
    self.assertEqual(s1.shape, (2, 4))
    self.assertEqual(r.shape, (2, 1))
    self.assertEqual(done.shape, (2, 1))

  def test_multi_step(self):
    self.memory = ReplayMemory(capacity=10, multi_step_n=2)
    for i in range(5):
      a = Transition([0, 1, 2, i], 0, [4, 5, 6, i*i], 1, False)
      self.memory.push(a)
    final = Transition([0, 1, 2, 10], 0, [4, 5, 6, 100], 10, True)
    self.memory.push(final)
    self.assertEqual(self.memory.memory[0].r, 2.9701)
    self.assertEqual(self.memory.memory[3].r, 11.791)
    self.assertEqual(self.memory.memory[4].r, 10.9)
    self.assertEqual(self.memory.memory[5].r, 10)

  def test_zero_step(self):
    self.memory = ReplayMemory(capacity=10, multi_step_n=0)
    for i in range(5):
      a = Transition([0, 1, 2, i], 0, [4, 5, 6, i*i], 1, False)
      self.memory.push(a)
    final = Transition([0, 1, 2, 10], 0, [4, 5, 6, 100], 10, True)
    self.memory.push(final)
    self.assertEqual(self.memory.memory[0].r, 1)
    self.assertEqual(self.memory.memory[3].r, 1)
    self.assertEqual(self.memory.memory[4].r, 1)
    self.assertEqual(self.memory.memory[5].r, 10)

if __name__ == "__main__":
  unittest.main()