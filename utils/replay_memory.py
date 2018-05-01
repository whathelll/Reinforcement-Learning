from collections import namedtuple, deque
import random
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])


class ReplayMemory(object):
  def __init__(self, capacity=1000, multi_step_n=0, multi_step_gamma=0.99):
    self.capacity = capacity
    self.memory = []
    self.n = multi_step_n + 1
    self.gamma = multi_step_gamma
    self.short_term_memory = deque(maxlen=self.n)
    self.position = 0

  def push(self, item):
    self.short_term_memory.append(item)
    if len(self.short_term_memory) == self.n:
      self._pop_short_term_memory()

    if item.done == True:
      while len(self.short_term_memory) > 0:
        self._pop_short_term_memory()

    # print("short term mem:", self.short_term_memory)
    # print(len(self.short_term_memory))

  def _pop_short_term_memory(self):
    reward = 0
    st_mem_len = len(self.short_term_memory)
    last_transition = self.short_term_memory[st_mem_len-1]
    for i in range(st_mem_len):
      reward += self.short_term_memory[i].r * (self.gamma ** i)
    # print("reward", reward)
    transition = self.short_term_memory.popleft()
    transition = Transition(transition.s, transition.a, last_transition.s_1, reward, last_transition.done)
    self._store_memory(transition)

  def _store_memory(self, item):
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
  def __init__(self, capacity, multi_step_n=0, multi_step_gamma=0.99, e=0.1, alpha=0.5):
    super().__init__(capacity=capacity, multi_step_n=multi_step_n, multi_step_gamma=multi_step_gamma)
    self.errors = []
    self.e = e
    self.alpha = alpha
    self.beta = 1

  def _store_memory(self, item):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.errors.append(None)
    self.errors[self.position] = 10000
    super()._store_memory(item)

  def sample(self, batch_size=2):
    probability = np.array(self.errors) ** self.alpha
    probability = probability / np.sum(probability)
    indices = np.random.choice(len(self.errors), size=batch_size, replace=False, p=probability)
    samples = np.take(self.memory, indices, axis=0)
    transitions = self._split_transition_samples(samples)
    #importance sampling weights
    # probability = np.take(probability, indices, axis=0)
    # weights = (batch_size * probability) ** self.beta
    # weights = weights / weights.max()
    # weights = np.expand_dims(weights, axis=1)
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

