import pytest 
import json

@pytest.fixture(scope="session")
def game_state_solo_start():
    with open("tests/data/game_state_solo_start.json") as f:
        yield json.load(f)