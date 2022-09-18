from src.battlesnake_gdoerksen.agentRewrite import AgentRewrite

def test_AgentRewrite_reductState(game_state_solo_start):
    agent = AgentRewrite()
    reduced_state = agent.reductState(game_state_solo_start)
    
    assert reduced_state == [0, 0, 0, 0, 0, 0, 0]
    