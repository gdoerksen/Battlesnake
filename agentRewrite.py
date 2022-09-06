# import python libraries
import typing
import numpy as np
import random
from collections import deque
import torch
from multiprocessing import Process

# import local libraries 
from model import Linear_QNet, QTrainer
from game_runner import BattleSnakeGameHandler
from game_runner import BattleSnakeURL

MAX_RANDOM_SEED = 1000000

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 # learning rate

EPSILON_NUMBER_OF_GAMES = 80

class AgentRewrite:
    def __init__(self):
        self.n_games_played = 0
        self.epsilon = 0    
        self.gamma = 0.9      # discount rate #TODO review discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size = 7, hidden_size = 256, output_size = 3)
        # TODO input size is the number of features in the state wich is 7 right now
        # Output size is 3 because we either go left, right or straight based on current direction
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)
        self.best_score = 0


    def moveAndTrain(self, game_state: typing.Dict)-> typing.Dict:
        is_game_over = False

        if game_state['turn'] == 0:
            self.state = game_state
            self.reduced_state = [1, 0, 0, 0, 1, 0, 0] #TODO this does not correspond with random choice
            self.move = [1, 0, 0]

            next_battle_move = random.choice(["up", "down", "left", "right"])
            print(f"MOVE {game_state['turn']}: {next_battle_move}")
            return {"move": next_battle_move}

        last_turn_state = self.state
        last_turn_move = self.move
        current_state = game_state

        last_turn_reduced_state = self.reduced_state
        current_reduced_state = self.reduceState(current_state)



        # we need to compute the reward for the last move
        reward, score = self.getReward(current_state, last_turn_state, is_game_over) 

        self.train_short_memory(last_turn_reduced_state, last_turn_move, reward, current_reduced_state, is_game_over)

        self.remember(last_turn_reduced_state, last_turn_move, reward, current_reduced_state, is_game_over)

        # update the state and move
        self.state = game_state
        self.reduced_state = current_reduced_state

        # next_move = self.getMove(current_reduced_state)
        next_move = self.getAction(current_reduced_state)
        self.move = next_move

        direction = self.getDirection(current_state)
        if next_move[0] == 1:
            next_battle_move = direction
        elif next_move[1] == 1:
            next_battle_move = self.getRightDirection(direction)
        elif next_move[2] == 1:
            next_battle_move = self.getLeftDirection(direction)

        print(f"MOVE {game_state['turn']}: {next_battle_move}")
        return {"move": next_battle_move}



    def getReward(self, game_state: typing.Dict, last_turn_state: typing.Dict, is_game_over: bool)-> typing.Dict:
        
        GAME_WIN_REWARD = 100  # we can only "win" the game if there are multiplie snakes and we are the last one standing
        GAME_LOSS_REWARD = -100
        FOOD_REWARD = 10
        IDLE_REWARD = 1
        # if we want to maxmimize staying alive, we should probably give a small reward for not dying

        if is_game_over:
            # check if we won
            # if game_state["you"]["health"] > 0: #TODO check that this is the correct way to check if we won
            #     return GAME_WIN_REWARD, True, game_state["turn"]  #TODO I think score is the number of turns
            # else:

            return GAME_LOSS_REWARD, game_state["turn"]

        else:
            if game_state["you"]["length"] > last_turn_state["you"]["length"]:
                #TODO check that this is the correct way to check if we ate food
                return FOOD_REWARD, game_state["turn"]

            #TODO add functions to reward the agent for moving towards the food
            return IDLE_REWARD, game_state["turn"] 


        
    def getDirection(self, game_state: typing.Dict)-> typing.Dict:
        my_body = game_state["you"]["body"]  # Coordinates of your "body"
        my_head = my_body[0]  # Coordinates of your head
        my_neck = my_body[1]  # Coordinates of your "neck"
        if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
            my_direction = "right"
        elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
            my_direction = "left"
        elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
            my_direction = "up"
        elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
            my_direction = "down"
        return my_direction

    def getRightDirection(self, direction: str)-> str:
        if direction == "up":
            return "right"
        elif direction == "right":
            return "down"
        elif direction == "down":
            return "left"
        elif direction == "left":
            return "up"

    def getLeftDirection(self, direction: str)-> str:
        if direction == "up":
            return "left"
        elif direction == "right":
            return "up"
        elif direction == "down":
            return "right"
        elif direction == "left":
            return "down"


    def reduceState(self, game_state: typing.Dict)-> typing.Dict:
        # currently we are using the Python Engineer method of reducing the state
        # this is bad 
        
        is_move_safe = {"up": True, "down": True, "left": True, "right": True}
        # reduce the state space to a smaller set of features
        
        my_body = game_state["you"]["body"]  # Coordinates of your "body"
        my_head = my_body[0]  # Coordinates of your head
        my_neck = my_body[1]  # Coordinates of your "neck"

        if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
            my_direction = "right"
            is_move_safe["left"] = False
        elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
            my_direction = "left"
            is_move_safe["right"] = False
        elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
            my_direction = "up"
            is_move_safe["down"] = False
        elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
            my_direction = "down"
            is_move_safe["up"] = False

        board_width = game_state['board']['width']
        board_height = game_state['board']['height']
        if my_head['x'] == 0:
            is_move_safe['left'] = False
        if my_head['x'] == board_width - 1:
            is_move_safe['right'] = False
        if my_head['y'] == 0:
            is_move_safe['down'] = False
        if my_head['y'] == board_height - 1:
            is_move_safe['up'] = False


        direction_up_bool = my_direction == "up"
        direction_down_bool = my_direction == "down"
        direction_left_bool = my_direction == "left"
        direction_right_bool = my_direction == "right"

        # state_list_bool = [is_move_safe["up"], is_move_safe["down"], is_move_safe["left"], is_move_safe["right"], 
        #             direction_up_bool, direction_down_bool, direction_left_bool, direction_right_bool]

        # state list bool is [danger straight, danger right, danger left, 
        #                     direction left, direction right, direction up, direction down,
        #                     food left, food right, food up, food down]

        # TODO this is so bad 
        if my_direction == "up":
            state_list_bool = [is_move_safe["up"], is_move_safe["right"], is_move_safe["left"], 
                            direction_left_bool, direction_right_bool, direction_up_bool, direction_down_bool]
        elif my_direction == "right":
            state_list_bool = [is_move_safe["right"], is_move_safe["down"], is_move_safe["up"], 
                            direction_left_bool, direction_right_bool, direction_up_bool, direction_down_bool]
        elif my_direction == "down":
            state_list_bool = [is_move_safe["down"], is_move_safe["left"], is_move_safe["right"], 
                            direction_left_bool, direction_right_bool, direction_up_bool, direction_down_bool]
        elif my_direction == "left":
            state_list_bool = [is_move_safe["left"], is_move_safe["up"], is_move_safe["down"], 
                            direction_left_bool, direction_right_bool, direction_up_bool, direction_down_bool]

        # TODO add the food and snake locations to the state

        return np.array(state_list_bool, dtype=int)

    def getAction(self, state):
        # state is the reduced state

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
            # print(prediction)
            move = torch.argmax(prediction).item()
            # print(move)
            final_move[move] = 1
            #TODO change into Battlesnake move

        # final move is one of:
        #   [1, 0, 0] # straight
        #   [0, 1, 0] # right 
        #   [0, 0, 1] # left
        return final_move


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

    def gameStarted(self, game_state: typing.Dict):
        is_game_over = False
        self.state = game_state
        #TODO initial state will be messed up because head is in the same place as the neck
        print(f"Game: {self.n_games_played} started")


    def gameEnded(self, game_state: typing.Dict):
        is_game_over = True

        last_turn_state = self.state
        last_turn_move = self.move
        current_state = game_state

        last_turn_reduced_state = self.reduced_state
        current_reduced_state = self.reduceState(current_state)

        # we need to compute the reward for the last move of the game
        reward, score = self.getReward(current_state, last_turn_state, is_game_over) 

        self.train_short_memory(last_turn_reduced_state, last_turn_move, reward, current_reduced_state, is_game_over)

        self.remember(last_turn_reduced_state, last_turn_move, reward, current_reduced_state, is_game_over)

        self.n_games_played += 1
        self.train_long_memory()

        if score > self.best_score:
            self.best_score = score
            self.model.save()

        print(f"Game: {self.n_games_played}, Score: {score}, Best score: {self.best_score}")

        # train short memory
        # remember
        # train long memory, plot result
        # game.reset() or go to gameStart 

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "DawnbringerX7",
        "color": "#333333",
        "head": "default",
        "tail": "default",
        }

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = AgentRewrite()

    from server import run_server
    server_args_dict = {"info": info, 
                        "start": agent.gameStarted, 
                        "move": agent.moveAndTrain, 
                        "end": agent.gameEnded}

    # snake_server = Process(target=run_server, args=(server_args_dict,))
    # snake_server.start()

    run_server(server_args_dict)

    # gameHandler = BattleSnakeGameHandler() 
    # snake1 = BattleSnakeURL(name="bellos", url="http://127.0.0.1:51689")
    # gameHandler.add_snakes(snake1)

    # while True: 
    #     random_seed = random.randint(0, MAX_RANDOM_SEED)
    #     gameHandler.set_random_seed(random_seed)
    #     gameHandler.start_game()



if __name__=="__main__":
    train()
