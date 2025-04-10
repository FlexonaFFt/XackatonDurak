import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import RANK_MAP, SUIT_MAP, STATE_MAP, MAX_HAND_SIZE, MAX_TABLE_SIZE

def encode_card(card_str):
    """Кодирование карты в числовой формат"""
    rank_map = {'9':0, '10':1, '11':2, '12':3, '13':4, '14':5}
    suit_map = {'S':0, 'C':1, 'D':2, 'H':3}
    
    rank = rank_map[card_str[:-1]]
    suit = suit_map[card_str[-1]]
    return rank * 4 + suit

def prepare_input(game_state, player_id):
    """Подготовка входных данных для модели из сырого JSON"""
    # Находим текущего игрока
    current_player = next(p for p in game_state['players'] if p['id'] == player_id)
    opponent = next(p for p in game_state['players'] if p['id'] != player_id)
    
    # Кодируем карты
    hand = [encode_card(c) for c in current_player['hand']]
    table = []
    for item in game_state['table']:
        table.append(encode_card(item['attack_card']['card']))
        if 'defend_card' in item:
            table.append(encode_card(item['defend_card']['card']))
    
    # Кодируем состояние игрока
    state_map = {'attack':0, 'defend':1, 'bat':2, 'pass':3, 
                'take':4, 'winner':5, 'durak':6}
    player_state = state_map[current_player['state']]
    
    # Создаем тензоры с паддингом
    hand_tensor = torch.zeros(6)  # Макс 6 карт
    hand_tensor[:len(hand)] = torch.tensor(hand)
    
    table_tensor = torch.zeros(12)  # Макс 6 пар карт
    table_tensor[:len(table)] = torch.tensor(table)
    
    return {
        'hand': hand_tensor.long(),
        'table': table_tensor.long(),
        'trump': torch.tensor([encode_card(game_state['trump'])]).long(),
        'player_state': torch.tensor([player_state]).long(),
        'deck_size': torch.tensor([len(game_state['deck'])])
    }