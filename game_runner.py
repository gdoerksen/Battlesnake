from typing import List
import subprocess
from pathlib import Path
from time import sleep

class BattleSnakeURL:
    def __init__(self, name, url):
        self.name = name
        self.url = url

class BattleSnakeGameHandler:
    def __init__(self, width:int=11,            # Width of Board (default 11)
                       height:int=11,           # Height of Board (default 11)
                       timeout:int=500,         # Timeout in ms (default 500)
                       sequential:bool=False,   # Use Sequential Processing (default False)
                       gametype:str='standard',
                       map:str='standard',
                       viewmap:bool=False,
                       color:bool=False,
                       seed:int=1656460409268690000, 
                       delay:int=500, #TODO turn delay does not have a default value
                       duration:int=0, #TODO duration does not have a default value
                       debug_requests:bool=False,
                       output:str='output.json',
                       browser:bool=False,
                       board_url:str='http://localhost:8080', #https://board.battlesnake.com
                       foodSpawnChance:int=15,
                       minimumFood:int=1,
                       hazardDamagePerTurn:int=14,
                       shrinkEveryNTurns:int=25
                       ):
        self.snake_class_list = []
        self.width = width
        self.height = height
        self.timeout = timeout
        self.sequential = sequential
        self.gametype = gametype
        self.map = map
        self.viewmap = viewmap
        self.color = color
        self.seed = seed
        self.delay = delay
        self.duration = duration
        self.debug_requests = debug_requests
        self.output = output
        self.browser = browser
        self.board_url = board_url
        self.foodSpawnChance = foodSpawnChance
        self.minimumFood = minimumFood
        
        #TODO hazard damage only occurs if there are hazards 
        self.hazardDamagePerTurn = hazardDamagePerTurn
        self.shrinkEveryNTurns = shrinkEveryNTurns

    #TODO what is sequential processing?

    #TODO --config string   config file (default is $HOME/.battlesnake.yaml)
    #TODO we can add a config file to setup games, instead of initializing the class with all the parameters

    def add_snakes(self, snake_class: BattleSnakeURL):
        self.snake_class_list.append(snake_class)

    def start_game(self):
        example_command = "battlesnake play -W 11 -H 11 --name bellos --url http://127.0.0.1:51689 -g solo"
        local_cli_path = Path(__file__).parent / 'local_battlesnake_cli' / 'battlesnake.exe'
        command = f"{local_cli_path} play -W {self.width} -H {self.height} --timeout {self.timeout} --name {self.snake_class_list[0].name} --url {self.snake_class_list[0].url} -g {self.gametype}"
        game_process = subprocess.run(command)
        # run the battlesnake.exe with the parameters and the snake urls

    def set_random_seed(self, random_seed:int):
        self.seed = random_seed


if __name__ == "__main__":
    snake1 = BattleSnakeURL(name="bellos", url="http://127.0.0.1:51689")

    gameHandler = BattleSnakeGameHandler()
    gameHandler.add_snakes(snake1)
    #TODO set random seed for the game

    for _ in range(100):
        gameHandler.start_game()
        sleep(1)

  

