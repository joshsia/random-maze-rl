The goal of the problem is to reach the goal in as few steps as possible.

The rules of the problem are:
- Maximum time for training is 10 minutes.

- The agent can take a maximum of 100 steps (i.e. if the agent does not reach the goal within 100 steps, it is considered a fail).

- The agent cannot only move through the light region (free space) and not the dark region (walls).

- If the agent tries to move through the dark region, the agent will remain in its current position.

- The length of one step taken by the agent cannot exceed 0.02. Otherwise, the agent will stay in the same position.

- Any form of memory is not allowed, other than by use of an experience replay buffer. For instance, networks that perform well cannot be saved and then loaded at a later time. Variables which store what the agent did in the previous timestep are also not allowed.

- An experience replay buffer is allowed, however, it cannot be used for anything other than training the Q-network. For instance, you cannot achieve any of the memory above by querying data in the replay buffer.

- You may not import anything from the `environment` module into the `agent` module to obtain information such as the goal state or the locations of obstacles.

- It can be assumed that the goal will always be to the right of the initial state.
