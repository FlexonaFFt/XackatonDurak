import pandas as pd
from datasets import load_dataset
from utils.preprocessing import encode_card
from utils.constants import STATE_MAP
from collections import defaultdict
import torch

def load_and_preprocess_data(dataset_name="neuronetties/durak", sample_size=None):
    dataset = load_dataset(dataset_name)
    print(f"Всего игр в датасете: {len(dataset['train'])}")
    
    df = pd.DataFrame(dataset['train'])
    if sample_size:
        df = df.sample(sample_size)
    
    processed_data = []
    for _, row in df.iterrows():
        try:
            current_player = row['players'][0]
            opponent = row['players'][1]
            
            sample = {
                'game_id': row['game_id'],
                'trump': encode_card(row['trump']),
                'player_state': STATE_MAP[current_player['state']],
                'hand': [encode_card(card) for card in current_player['hand']],
                'opponent_hand_size': len(opponent['hand']),
                'deck_size': len(row['deck']),
                'table_cards': process_table_cards(row['table']),
                'timestamp': row['timestamp'],
                'winner': row['winner']
            }
            
            sample.update(extract_labels(row, current_player))
            processed_data.append(sample)
            
        except Exception as e:
            print(f"Ошибка обработки игры {row['game_id']}: {str(e)}")
    
    return pd.DataFrame(processed_data)

def process_table_cards(table):
    """Обработка карт на столе"""
    table_cards = []
    for item in table:
        table_cards.append(encode_card(item['attack_card']['card']))
        if 'defend_card' in item:
            table_cards.append(encode_card(item['defend_card']['card']))
    return table_cards

def extract_labels(game_data, player):
    """Извлечение меток для обучения"""
    labels = {
        'value_target': 1.0 if player['id'] == game_data['winner'] else -1.0,
        'policy_target': 0  # Заглушка - реальные метки нужно определить по логике игры
    }
    
    # Здесь должна быть ваша логика определения правильного действия
    # Например, анализ следующего состояния игры
    
    return labels

def create_dataloader(df, batch_size=32):
    """Создание DataLoader для обучения"""
    class DurakDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            sample = self.data.iloc[idx]
            
            # Подготовка входных тензоров
            hand_tensor = torch.zeros(6)  # MAX_HAND_SIZE
            hand_tensor[:len(sample['hand'])] = torch.tensor(sample['hand'])
            
            table_tensor = torch.zeros(12)  # MAX_TABLE_SIZE
            table_tensor[:len(sample['table_cards'])] = torch.tensor(sample['table_cards'])
            
            return {
                'inputs': {
                    'hand': hand_tensor.long(),
                    'table': table_tensor.long(),
                    'trump': torch.tensor([sample['trump']]).long(),
                    'player_state': torch.tensor([sample['player_state']]).long(),
                    'deck_size': torch.tensor([sample['deck_size']]).float()
                },
                'targets': {
                    'policy': torch.tensor(sample['policy_target']).long(),
                    'value': torch.tensor(sample['value_target']).float()
                }
            }
    
    return torch.utils.data.DataLoader(
        DurakDataset(df),
        batch_size=batch_size,
        shuffle=True
    )

def process_game_sequences(dataset, player_id="our_bot"):
    """
    Обрабатывает последовательности ходов в играх, извлекая:
    - признаки состояния игры
    - метки правильных действий
    - ценность позиции
    
    Args:
        dataset: Загруженный датасет с играми
        player_id: ID нашего бота в данных
        
    Returns:
        List[Dict]: Список готовых примеров для обучения
    """
    processed_samples = []
    
    # Группируем снимки по game_id
    games = defaultdict(list)
    for item in dataset:
        snapshot = json.loads(item['snapshot'])
        games[snapshot['game_id']].append(snapshot)
    
    # Обрабатываем каждую игру отдельно
    for game_id, snapshots in games.items():
        # Сортируем по timestamp
        snapshots.sort(key=lambda x: x['timestamp'])
        
        # Определяем победителя (только для value target)
        winner = snapshots[-1]['winner']
        
        # Обрабатываем каждый переход состояния
        for i in range(len(snapshots)-1):
            current_state = snapshots[i]
            next_state = snapshots[i+1]
            
            # Проверяем, что это наш ход
            current_player = next(p for p in current_state['players'] if p['id'] == player_id)
            if current_player['state'] in ['wait', 'winner', 'durak']:
                continue
                
            # Извлекаем признаки (без информации о будущем!)
            features = extract_features(current_state, player_id)
            
            # Извлекаем метку действия
            action_label = extract_action_label(current_state, next_state, player_id)
            
            # Вычисляем ценность позиции
            value_label = 1.0 if winner == player_id else -1.0
            
            processed_samples.append({
                'features': features,
                'action_label': action_label,
                'value_label': value_label,
                'game_id': game_id,
                'timestamp': current_state['timestamp']
            })
    
    return processed_samples

def extract_features(game_state, player_id):
    """Извлекает признаки из текущего состояния игры"""
    current_player = next(p for p in game_state['players'] if p['id'] == player_id)
    opponent = next(p for p in game_state['players'] if p['id'] != player_id)
    
    # Основные признаки
    features = {
        'hand': [encode_card(c) for c in current_player['hand']],
        'opponent_hand_size': len(opponent['hand']),
        'trump': encode_card(game_state['trump']),
        'player_state': STATE_MAP[current_player['state']],
        'deck_size': len(game_state['deck']),
        'table': process_table_cards(game_state['table']),
        'game_type': game_state['game_rules']['game_type']
    }
    
    # Дополнительные engineered features
    features['trump_in_hand'] = any(c[-1] == game_state['trump'][-1] for c in current_player['hand'])
    features['can_transfer'] = can_transfer(current_player, game_state['table'])
    
    return features

def extract_action_label(current_state, next_state, player_id):
    """Определяет какое действие было совершено между состояниями"""
    current_player = next(p for p in current_state['players'] if p['id'] == player_id)
    
    # Анализируем изменения
    if current_player['state'] == 'attack':
        # Находим новую карту на столе
        new_card = find_new_attack(current_state['table'], next_state['table'])
        return f"attack_{new_card}" if new_card else "bat"
    
    elif current_player['state'] == 'defend':
        if next_state['players'][0]['state'] == 'take':
            return "take"
        new_defense = find_new_defense(current_state['table'], next_state['table'])
        return f"defend_{new_defense}" if new_defense else "take"
    
    return "wait"