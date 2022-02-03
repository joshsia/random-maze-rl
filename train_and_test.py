import time
import numpy as np

from random_environment import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":

    tests = 1
    seeds = []
    success = 0
    fail = 0
    for i in range(tests):
        # This determines whether the environment will be displayed on each each step.
        display_on = True

        # Create a random seed, which will define the environment
        random_seed = int(time.time())

        # Hard environment seeds:
        # [1606630931, 1606632731, 1606633331, 1606414252, 1606480834]
        random_seed = 1606480834
        np.random.seed(random_seed)

        # Create a random environment
        environment = Environment(magnification=500)

        # Create an agent
        agent = Agent()

        # Get the initial state
        state = environment.init_state
        show = 0

        # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
        start_time = time.time()
        #end_time = start_time + 540
        end_time = start_time + 600

        # Train the agent, until the time is up
        while time.time() < end_time:
            # If the action is to start a new episode, then reset the state
            if agent.has_finished_episode():
                state = environment.init_state
            # Get the state and action from the agent
            action = agent.get_next_action(state)
            # Get the next state and the distance to the goal
            next_state, distance_to_goal = environment.step(state, action)
            # Return this to the agent
            agent.set_next_state_and_distance(next_state, distance_to_goal)
            # Set what the new state is
            state = next_state
            # Optionally, show the environment
            if display_on and (show < 3):
            #if display_on:
                show += 1
                environment.show(state)

        # Test the agent for 100 steps, using its greedy policy

        #pause = input('\nReady to see greedy policy?')
        state = environment.init_state
        has_reached_goal = False
        for step_num in range(100):
            action = agent.get_greedy_action(state)
            next_state, distance_to_goal = environment.step(state, action)
            # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
            if distance_to_goal < 0.03:
                has_reached_goal = True
                break
            state = next_state
            if display_on:
                environment.show(state)

        # Print out the result
        print("\nTrial {}".format(i))
        if has_reached_goal:
            success += 1
            print('Reached goal in ' + str(step_num) + ' steps.')
        else:
            fail += 1
            seeds.append(random_seed)
            print('Did not reach goal. Final distance = ' + str(distance_to_goal))
    
    print("Seeds: {}".format(seeds))
    print("Success: {}; Failed: {}".format(success, fail))
