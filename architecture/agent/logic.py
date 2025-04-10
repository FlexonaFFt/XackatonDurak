import torch
import torch.nn.functional as F

from model.model import DurakModel
from utils.preprocessing import prepare_input

class DurakAgent:
    def __init__(self, model_path, player_id='bot1', device='cuda'):
        self.model = DurakModel().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device
        self.player_id = player_id
        
    def get_action(self, game_state):
        inputs = prepare_input(game_state, self.player_id)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits, _ = self.model(inputs)
            probs = F.softmax(logits, dim=-1)
        
        return self._map_action(probs.argmax().item(), game_state)
    
    def _map_action(self, action_idx, game_state):
        current_player = next(p for p in game_state['players'] if p['id'] == self.player_id)
        
        if current_player['state'] == 'attack':
            return self._attack_action(action_idx, game_state)
        elif current_player['state'] == 'defend':
            return self._defend_action(action_idx, game_state)
        else:
            return self._state_action(action_idx)
    
    def _attack_action(self, action_idx, game_state):
        hand = game_state['players'][0]['hand']
        return {"type": "attack", "move": hand[action_idx % len(hand)]}
    
    def _defend_action(self, action_idx, game_state):
        table = game_state['table']
        hand = game_state['players'][0]['hand']
        
        if table and action_idx % 2 == 1:
            return {
                "type": "defend",
                "move": [[table[-1]['attack_card']['user_id'], table[-1]['attack_card']['card']],
                        [self.player_id, hand[action_idx % len(hand)]]]
            }
        return {"type": "wait"}
    
    def _state_action(self, action_idx):
        actions = ["pass", "bat", "take"]
        return {"type": "state", "state": actions[action_idx % 3]}