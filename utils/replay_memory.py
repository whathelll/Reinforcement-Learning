from collections import namedtuple
import random
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])


class ReplayMemory(object):
  def __init__(self, capacity=1000):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, item):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = item
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size=2):
    samples = random.sample(self.memory, batch_size)
    return self._split_transition_samples(samples)

  """splits a list of Transitions into [s, a, s_1, r, done]"""
  def _split_transition_samples(self, samples):
    batched = Transition(*zip(*samples))
    s = np.array(list(batched.s))
    a = np.expand_dims(np.array(list(batched.a)), axis=1)
    s_1 = np.array(list(batched.s_1))
    r = np.expand_dims(np.array(list(batched.r)), axis=1)
    done = np.expand_dims(np.array(list(batched.done)), axis=1)
    return [s, a, s_1, r, done]

  def __len__(self):
    return len(self.memory)

  def __str__(self):
    result = []
    for i in range(self.__len__()):
      result.append(self.memory[i].__str__() + " \n")
    return "".join(result)


# Proportional: I don't understand SumTrees
class PrioritisedReplayMemory(ReplayMemory):
  def __init__(self, capacity, e=0.1, alpha=0.5):
    super().__init__(capacity)
    self.errors = []
    self.e = e
    self.alpha = alpha

  def push(self, item):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.errors.append(None)
    self.errors[self.position] = 10000
    super().push(item)

  def sample(self, batch_size=2):
    probability = np.array(self.errors) ** self.alpha
    probability = probability / np.sum(probability)
    indices = np.random.choice(len(self.errors), size=batch_size, replace=False, p=probability)
    samples = np.take(self.memory, indices, axis=0)
    transitions = self._split_transition_samples(samples)
    return transitions, indices

  def update(self, indices, errors):
    errors = np.absolute(errors)
    for i in range(len(indices)):
      self.errors[indices[i]] = errors[i] + self.e

  def __str__(self):
    result = []
    for i in range(self.__len__()):
      result.append(self.memory[i].__str__() + " error:" + self.errors[i].__str__() + " \n")
    return "".join(result)
#
# # Proportional: I don't understand SumTrees
# class PrioritisedReplayMemory(object):
#   def __init__(self, capacity, e=0.1, alpha=0.5):
#     self.capacity = capacity
#     self.memory = []
#     self.errors = []
#     self.position = 0
#     self.e = e
#     self.alpha = alpha
#
#   def push(self, item):
#     """Saves a transition."""
#     if len(self.memory) < self.capacity:
#       self.memory.append(None)
#       self.errors.append(None)
#     self.memory[self.position] = item
#     self.errors[self.position] = 10000
#     self.position = (self.position + 1) % self.capacity
#
#   def sample(self, batch_size):
#     probability = np.array(self.errors) ** self.alpha
#     probability = probability / np.sum(probability)
#
#     indices = np.random.choice(len(self.errors), size=batch_size, replace=False, p=probability)
#     out = np.take(self.memory, indices, axis=0)
#     batched = Transition(*zip(*out))
#     s = np.array(list(batched.s))
#     a = np.expand_dims(np.array(list(batched.a)), axis=1)
#     s_1 = np.array(list(batched.s_1))
#     r = np.expand_dims(np.array(list(batched.r)), axis=1)
#     done = np.expand_dims(np.array(list(batched.done)), axis=1)
#     return [s, a, s_1, r, done], indices
#
#   def update(self, indices, errors):
#     errors = np.absolute(errors)
#     for i in range(len(indices)):
#       self.errors[indices[i]] = errors[i] + self.e
#
#   def __len__(self):
#     return len(self.memory)
#
#   def __str__(self):
#     result = []
#     for i in range(self.__len__()):
#       result.append(self.memory[i].__str__() + " error:" + self.errors[i].__str__() + " \n")
#     return "".join(result)
#
#
