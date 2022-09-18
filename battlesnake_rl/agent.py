# import python libraries
import torch 
import random
import numpy as np

from collections import deque

# import local libraries
from battlesnake_rl.game_runner import BattleSnakeGameHandler
from battlesnake_rl.game_runner import BattleSnakeURL
from battlesnake_rl.model import Linear_QNet, QTrainer

MAX_RANDOM_SEED = 1000000

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 # learning rate

EPSILON_NUMBER_OF_GAMES = 80

class Agent:
    def __init__(self):
        self.n_games_played = 0
        self.epsilon = 0    
        self.gamma = 0.9      # discount rate #TODO review discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size = 11, hidden_size = 256, output_size = 3)
        # TODO input size is the number of features in the state wich is 8 right now
        # Output size is 3 because we either go left, right or straight based on current direction
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

        # TODO: model, trainer

    def get_state(self, state):
        pass

    def remember(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, is_game_overs = zip(*mini_sample) # unzip the tuples (for loop?)
        self.trainer.train_step(states, actions, rewards, next_states, is_game_overs)

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.trainer.train_step(state, action, reward, next_state, is_game_over)

    def get_action(self, state):
        # [1, 0, 0] # straight
        # [0, 1, 0] # right 
        # [0, 0, 1] # left

        # random moves: tradeoff between exploration and exploitation
        self.epsilon = EPSILON_NUMBER_OF_GAMES - self.n_games_played
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            #TODO change into Battlesnake move

        else: 
            # get prediction from the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            #TODO change into Battlesnake move

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()

    # agent = Agent()
    # this should start the snake server? 

    # from server import run_server
    # myBattleSnake = BattleSnake("DawnbringerX7", "#333333", "default", "default")

    # run_server({"info": myBattleSnake.info, 
    #             "start": myBattleSnake.start, 
    #             "move": myBattleSnake.move, 
    #             "end": myBattleSnake.end})
    
    # setup game environment
    gameHandler = BattleSnakeGameHandler() 
    snake1 = BattleSnakeURL(name="bellos", url="http://127.0.0.1:51689")
    gameHandler.add_snakes(snake1)

    random_seed = random.randint(0, MAX_RANDOM_SEED)
    gameHandler.set_random_seed(random_seed)

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
            
            #game.reset()
            # set a new random seed for the next game
            random_seed = random.randint(0, MAX_RANDOM_SEED)
            gameHandler.set_random_seed(random_seed)
            
            agent.n_games_played += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save()
            
            print(f"Game: {agent.n_games_played}, Score: {score}, Best score: {best_score}")
            #TODO make plots 


            from helper import plot #TODO (see 4th video)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games_played
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__=="__main__":
    train()