import typing
import numpy as np

class AgentRewrite:

    def moveAndTrain(self, game_state: typing.Dict)-> typing.Dict:
        if self.turn == 0:
            self.doARandomMove()

        last_turn_state = self.state
        last_turn_move = self.move
        current_state = game_state

        last_turn_reduced_state = self.reduced_state
        current_reduced_state = self.reduceState(current_state)

        # we need to compute the reward for the last move
        reward, is_game_over, score = self.getReward(game_state) #, last_turn_state, last_turn_move)
        # TODO where is the reward calculated? either above in play_step or below in train_short_memory

        state_new = game_state
        final_move = last_turn_move
        self.train_short_memory(last_turn_state, final_move, reward, state_new, is_game_over)

        self.remember(last_turn_state, final_move, reward, state_new, is_game_over)

        # update the state and move
        self.state = game_state

        next_move = self.getMove(game_state)
        self.move = next_move

        print(f"MOVE {game_state['turn']}: {next_move}")
        return {"move": next_move}


    def getMove(self, game_state: typing.Dict)-> typing.Dict:
        pass

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

        state_list_bool = [is_move_safe["up"], is_move_safe["down"], is_move_safe["left"], is_move_safe["right"], 
                            direction_up_bool, direction_down_bool, direction_left_bool, direction_right_bool]
        # TODO add the food and snake locations to the state

        return np.array(state_list_bool, dtype=int)


    def getReward(self, game_state: typing.Dict)-> typing.Dict:
        pass

    def gameEnded(self, game_state: typing.Dict)-> typing.Dict:
        
        # train short memory
        # remember
        # train long memory, plot result
        # game.reset() or go to gameStart 