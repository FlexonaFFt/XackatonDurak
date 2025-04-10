import pandas as pd
import numpy as np
from datasets import load_dataset
from utils.constants import STATE_MAP
from collections import defaultdict
import torch
import json 

def preprocess_dataset(raw_data):
    """
    Основная функция предобработки данных
    
    Args:
        raw_data: список JSON-объектов с данными игр
        
    Returns:
        processed_games: словарь с обработанными играми, структурированными по game_id
        player_stats: статистика по игрокам
    """
    # 1. Группировка данных по game_id
    games_by_id = defaultdict(list)
    for snapshot in raw_data:
        games_by_id[snapshot['game_id']].append(snapshot)
    
    # 2. Сортировка снимков внутри каждой игры по timestamp
    for game_id in games_by_id:
        games_by_id[game_id].sort(key=lambda x: x['timestamp'])
    
    processed_games = {}
    player_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_moves': 0})
    
    # 3. Обработка каждой игры
    for game_id, snapshots in games_by_id.items():
        game_data = {
            'game_type': snapshots[0]['game_rules']['game_type'],
            'trump': snapshots[0]['trump'],
            'winner': snapshots[-1]['winner'],
            'timesteps': [],
            'initial_deck': snapshots[0]['deck'].copy(),
            'players': [p['id'] for p in snapshots[0]['players']]
        }
        
        # 4. Обработка каждого шага в игре
        for i, snapshot in enumerate(snapshots):
            timestep = {
                'timestamp': snapshot['timestamp'],
                'deck_size': len(snapshot['deck']),
                'bat': snapshot['bat'],
                'table': snapshot['table'],
                'player_states': {},
                'valid_moves': {}
            }
            
            # 5. Обработка состояния каждого игрока
            for player in snapshot['players']:
                player_id = player['id']
                timestep['player_states'][player_id] = {
                    'state': player['state'],
                    'hand': player['hand'],
                    'hand_size': len(player['hand'])
                }
                
                # 6. Определение допустимых ходов для каждого игрока
                if player['state'] == 'attack':
                    timestep['valid_moves'][player_id] = get_valid_attacks(player, snapshot)
                elif player['state'] == 'defend':
                    timestep['valid_moves'][player_id] = get_valid_defenses(player, snapshot)
                else:
                    timestep['valid_moves'][player_id] = get_state_actions(player['state'])
            
            game_data['timesteps'].append(timestep)
            
            # 7. Сбор статистики по игрокам
            if i > 0:  
                for player in snapshot['players']:
                    player_stats[player['id']]['total_moves'] += 1
        
        # 8. Обновление статистики побед/поражений
        winner = game_data['winner']
        for player_id in game_data['players']:
            if player_id == winner:
                player_stats[player_id]['wins'] += 1
            else:
                player_stats[player_id]['losses'] += 1
        
        processed_games[game_id] = game_data
    
    return processed_games, player_stats

def get_valid_attacks(player, snapshot):
    valid_moves = []
    hand = player['hand']
    if not snapshot['table']:
        return hand
    
    table_ranks = {card['attack_card']['card'][:-1] for card in snapshot['table']}
    valid_moves = [card for card in hand if card[:-1] in table_ranks]
    
    return valid_moves

def get_valid_defenses(player, snapshot):
    valid_moves = []
    hand = player['hand']
    trump = snapshot['trump'][-1]
    
    for attack in snapshot['table']:
        if 'defend_card' not in attack:
            attack_card = attack['attack_card']['card']
            attack_rank = attack_card[:-1]
            attack_suit = attack_card[-1]
            
            for card in hand:
                card_suit = card[-1]
                card_rank = card[:-1]
                
                # Если это козырь и атакующая карта не козырь
                if card_suit == trump and attack_suit != trump:
                    valid_moves.append((attack_card, card))
                # Если масть совпадает и достоинство больше
                elif card_suit == attack_suit and RANKS[card_rank] > RANKS[attack_rank]:
                    valid_moves.append((attack_card, card))
    
    return valid_moves

def get_state_actions(state):
    if state in ['bat', 'pass', 'take']:
        return [state]
    return []

RANKS = {'9': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5}