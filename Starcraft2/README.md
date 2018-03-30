# Experiments in Deep Reinforcement Learning


## Starcraft 2 DQN Agent
To run
```bash
export SC2PATH=/..../StarCraftII
PYTHONPATH=. python sc2_agents/BaseTrainer.py --map=MoveToBeacon
```

to train
```bash
PYTHONPATH=. python sc2_agents/BaseTrainer.py --map=MoveToBeacon --train=True
```