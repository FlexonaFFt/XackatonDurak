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





def prepare_training_data(processed_games):
    """
    Подготавливает данные для обучения модели
    
    Args:
        processed_games: обработанные данные игр
        
    Returns:
        X: признаки для обучения
        y: целевые переменные
        metadata: дополнительная информация о примерах
    """
    X = []
    y = []
    metadata = []
    
    for game_id, game in processed_games.items():
        game_type = game['game_type']
        trump = game['trump']
        
        for i in range(len(game['timesteps']) - 1):
            current_step = game['timesteps'][i]
            next_step = game['timesteps'][i + 1]
            
            # Для каждого игрока в текущем состоянии
            for player_id, player_state in current_step['player_states'].items():
                # Создаем вектор признаков
                features = create_feature_vector(player_id, current_step, game_type, trump)
                
                # Определяем правильный ход (на основе следующего состояния)
                correct_move = determine_correct_move(player_id, current_step, next_step)
                
                if features and correct_move:
                    X.append(features)
                    y.append(correct_move)
                    metadata.append({
                        'game_id': game_id,
                        'player_id': player_id,
                        'timestamp': current_step['timestamp']
                    })
    
    return np.array(X), np.array(y), metadata

def create_feature_vector(player_id, timestep, game_type, trump):
    """Создает вектор признаков для текущего состояния игрока"""
    # 1. Информация о руке игрока
    hand = timestep['player_states'][player_id]['hand']
    hand_features = encode_hand(hand, trump)
    
    # 2. Информация о столе
    table_features = encode_table(timestep['table'], trump)
    
    # 3. Информация о бито
    bat_features = encode_bat(timestep['bat'], trump)
    
    # 4. Информация о колоде
    deck_features = [timestep['deck_size'] / 24.0]  # Нормализованный размер колоды
    
    # 5. Информация о состоянии игрока
    state = timestep['player_states'][player_id]['state']
    state_features = encode_state(state)
    
    # 6. Информация о типе игры (классическая/переводная)
    game_type_feature = [game_type]
    
    # Объединяем все признаки
    feature_vector = (
        hand_features +
        table_features +
        bat_features +
        deck_features +
        state_features +
        game_type_feature
    )
    
    return feature_vector

def determine_correct_move(player_id, current_step, next_step):
    """Определяет правильный ход на основе следующего состояния"""
    # Здесь должна быть логика определения, какой ход был сделан игроком
    # между current_step и next_step
    # Это может быть сложно и требует анализа изменений в состоянии
    
    # Упрощенная версия - берем первый возможный ход (нужно доработать)
    valid_moves = current_step['valid_moves'].get(player_id, [])
    if valid_moves:
        return encode_move(valid_moves[0])
    return None