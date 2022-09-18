# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import json

class BattleSnake:
    def __init__(self, name, color, head, tail):
        self.name = name
        self.color = color
        self.head = head
        self.tail = tail

    def info(self) -> typing.Dict:
        return {
            "apiversion": "1",
            "author": self.name,
            "color": self.color,
            "head": self.head,
            "tail": self.tail,
        }

    def start(self, game_state: typing.Dict):
        print("GAME START")

        

    def end(self, game_state: typing.Dict):
        print("GAME END")

    def move(self, game_state: typing.Dict)-> typing.Dict:
        is_move_safe = {"up": True, "down": True, "left": True, "right": True}

        # We've included code to prevent your Battlesnake from moving backwards
        my_body = game_state["you"]["body"]  # Coordinates of your "body"
        my_head = my_body[0]  # Coordinates of your head
        my_neck = my_body[1]  # Coordinates of your "neck"

        my_length = game_state["you"]["length"]  # Length of your Battlesnake
        my_health = game_state["you"]["health"]  # Health of your Battlesnake

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

        # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
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

        # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
        # my_body is a list of dictionaries, each dictionary has a key 'x' and 'y' 
        above = {'x': my_head['x'], 'y': my_head['y'] + 1}
        below = {'x': my_head['x'], 'y': my_head['y'] - 1}
        left = {'x': my_head['x'] - 1, 'y': my_head['y']}
        right = {'x': my_head['x'] + 1, 'y': my_head['y']}
        if above in my_body:
            is_move_safe['up'] = False
        if below in my_body:
            is_move_safe['down'] = False
        if left in my_body:
            is_move_safe['left'] = False
        if right in my_body:
            is_move_safe['right'] = False


        # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
        # opponents = game_state['board']['snakes']

        # Are there any safe moves left?
        safe_moves = []
        for move, isSafe in is_move_safe.items():
            if isSafe:
                safe_moves.append(move)

        if len(safe_moves) == 0:
            print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
            return {"move": "down"}

        # Choose a random move from the safe ones
        next_move = random.choice(safe_moves)

        # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
        # food = game_state['board']['food']

        print(f"MOVE {game_state['turn']}: {next_move}")
        return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from battlesnake_rl.server import run_server
    myBattleSnake = BattleSnake("DawnbringerX7", "#333333", "default", "default")

    run_server({"info": myBattleSnake.info, 
                "start": myBattleSnake.start, 
                "move": myBattleSnake.move, 
                "end": myBattleSnake.end})

"""
The board can be represented as a graph with nodes being the coordinates 
of the board and edges being the possible moves.

Will taking a move lead to a dead end?
   if it looks like a cave 

We could just make a rule not to make a cyclic graph 

Look for longest current path?

bread first search starting from head?

compute the shortest path to food?
and our enemies shortest path to food and compare?

enemy snake data:
    growth rate
    did their previous move decrease their manhattan distance toward previous food?



"""


"""
Environment:
    board 
    - food locations
    - enemy snake locations
    - hazards
    - board width
    - board height 
    
    myself
    - myHealth
    - myLocation
    - myLength

Output:
    left, right, up, down 

"""

"""
Deep Q Learning

* initialize Q value

* choose action - model.predict(state)
* perform action
* measure reward
* update Q value (train model)
* go back to choose action

"""



"""
Python Engineer vids

State (11 values)
- danger direction straight, right or left [0, 0, 0]
- snake direction left, right, up and down [ 0, 1, 0, 0]
- food left, food right, food up, food down [0, 1, 0, 1]
    * not right next to, but just directional

"""