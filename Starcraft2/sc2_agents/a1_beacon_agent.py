# copy of PYSC2 random agent


import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SELECT_POINT = actions.FUNCTIONS.select_point.id

class BeaconAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(BeaconAgent, self).step(obs)

    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      # return actions.FunctionCall(_NO_OP, [])
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      target = [int(friendly_x.mean()), int(friendly_y.mean())]
      return actions.FunctionCall(_SELECT_POINT, [[0], target])

    # function_id = numpy.random.choice(obs.observation["available_actions"])

    # print(function_id)

    # args = [[numpy.random.randint(0, size) for size in arg.sizes]
    #         for arg in self.action_spec.functions[function_id].args]
    # return actions.FunctionCall(function_id, args)
    # observation =
