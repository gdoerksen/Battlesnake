import torch 
import random
import numpy as np

from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:
    def __init__(self):
        self.n_games_played = 0
        self.epsilon = 0    
        self.gamma = 0      # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        # TODO: model, trainer

    def get_state(self, state):
        pass

    def remember(self, state, action, reward, next_state, is_game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = BattleSnakeGame() #TODO - implement this

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, is_game_over, score = game.play_step(final_move) #TODO will be a call to MOVE
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, is_game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, is_game_over)

        if is_game_over:
            # train long memory, plot result
            game.reset()
            agent.n_games_played += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                #TODO: agent.model.save()
            
            print(f"Game: {agent.n_games_played}, Score: {score}, Best score: {best_score}")
            #TODO make plots 


if __name__=="__main__":
    train()