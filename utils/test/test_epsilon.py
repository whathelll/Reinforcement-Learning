import unittest
from utils.epsilon import Epsilon

class TestEpsilon(unittest.TestCase):
  def setUp(self):
    self.epsilon = Epsilon(start=1.0, end=0.01, update_increment=0.01)

  def test_value(self):
    self.epsilon.value()
    self.assertEqual(self.epsilon.value(), 1)

  def test_increment(self):
    self.epsilon.increment()\
      .increment()\
      .increment()
    self.assertEqual(self.epsilon.value(), 0.97)
    self.epsilon.increment(99)
    self.assertEqual(self.epsilon.value(), 0.01)

  def test_isTraing(self):
    self.epsilon.isTraining = False
    self.assertEqual(self.epsilon.value(), 0.0)

    self.epsilon.isTraining = True
    self.assertEqual(self.epsilon.value(), 1.0)


if __name__ == '__main__':
  unittest.main()