# RL-testing-ground
Evaluation of different RL decision methods against the K-armed bandits test

## Running the code
Needs Matplotlib and numpy. 
Values for the number of episodes, arms (k), and iterations can be changed in lines 4,5, and 6 of the file.

## Current Decision methods
- Elipson greedy. A threshold is given for exploration (random action) or explotation (highest valued from current data)
- Elipson decay. Like elipson greedy but the threshold decays exponentially with experience
- Upper confidence bound. Chooses the maximal action, but adding value based on how much experience it has with each arm (more uncertain, more optimistic).

