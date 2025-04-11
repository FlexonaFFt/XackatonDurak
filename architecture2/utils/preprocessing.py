import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import defaultdict
import json
from datasets import load_dataset

class DurakDataset(Dataset):
    def __init__(self, hf_dataset="neuronetties/durak", split="train", limit=None):
        """
        Полностью переработанный загрузчик данных для игры Дурак
        
        Args:
            hf_dataset: имя датасета на HuggingFace
            split: "train" или "validation"
            limit: ограничение количества примеров
        """
        # Инициализация кодировщиков
        self.card_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self._init_encoders()
        
        # Загрузка данных
        self.raw_data = load_dataset(hf_dataset, split=split)
        self.data = list(self.raw_data.select(range(limit))) if limit else list(self.raw_data)
        
        print(f"Successfully loaded {len(self.data)} raw game states")
        
        # Группировка и подготовка данных
        self.games = self._organize_games()
        self.samples = self._create_training_samples()
        
        if not self.samples:
            raise ValueError("Failed to create training samples. Please check your data structure.")
        
        print(f"Successfully prepared {len(self.samples)} training samples")

    def _init_encoders(self):
        """Инициализация кодировщиков карт и действий"""
        # Все возможные карты
        card_values = ['9','10','11','12','13','14']
        card_suits = ['S','C','D','H']
        cards = [f"{value}{suit}" for value in card_values for suit in card_suits]
        self.card_encoder.fit(cards + ['None'])  # None для действий без карты
        
        # Все возможные действия
        actions = ['attack', 'defend', 'take', 'pass', 'bat']
        self.action_encoder.fit(actions)

    def _organize_games(self):
        """Группирует сырые данные по играм с проверкой валидности"""
        games = defaultdict(list)
        
        for state in self.data:
            try:
                # Базовые проверки структуры
                if not isinstance(state, dict):
                    continue
                    
                if 'players' not in state or len(state['players']) < 2:
                    continue
                    
                if 'game_id' not in state:
                    continue
                    
                games[state['game_id']].append(state)
            except:
                continue
        
        # Фильтрация неполных игр
        valid_games = {}
        for game_id, states in games.items():
            try:
                # Игра должна содержать хотя бы 2 состояния
                if len(states) < 2:
                    continue
                    
                # Сортируем состояния по timestamp или порядку
                sorted_states = sorted(
                    states,
                    key=lambda x: x.get('timestamp', 0)
                )
                
                # Проверяем наличие победителя в последнем состоянии
                last_state = sorted_states[-1]
                if 'winner' not in last_state:
                    continue
                    
                valid_games[game_id] = sorted_states
            except:
                continue
                
        print(f"Found {len(valid_games)} valid games out of {len(games)}")
        return valid_games

    def _create_training_samples(self):
        """Создает обучающие примеры из валидных игр"""
        samples = []
        
        for game_id, states in self.games.items():
            try:
                winner_id = states[-1]['winner']
                
                for i in range(1, len(states)):
                    prev_state = states[i-1]
                    current_state = states[i]
                    
                    # Извлекаем признаки
                    features = self._extract_game_features(prev_state)
                    if features is None:
                        continue
                        
                    # Определяем действие
                    action = self._determine_action(prev_state, current_state)
                    if action is None:
                        continue
                        
                    # Определяем игрока
                    current_player_id = current_state['players'][0]['id']
                    is_winner_move = (current_player_id == winner_id)
                    
                    samples.append({
                        'features': features,
                        'action': action,
                        'is_winner': is_winner_move,
                        'game_state': prev_state
                    })
            except Exception as e:
                print(f"Skipping game {game_id} due to error: {str(e)}")
                continue
                
        return samples

    def _extract_game_features(self, state):
        """Векторизация состояния игры"""
        try:
            # 1. Козырь
            trump = state['trump']
            trump_suit = trump[-1]
            trump_encoded = [0, 0, 0, 0]
            suits = ['S', 'C', 'D', 'H']
            trump_encoded[suits.index(trump_suit)] = 1
            
            # 2. Карты в руке (текущий игрок)
            hand_encoded = [0] * len(self.card_encoder.classes_)
            for card in state['players'][0]['hand']:
                idx = self.card_encoder.transform([card])[0]
                hand_encoded[idx] = 1
                
            # 3. Карты на столе
            table_encoded = [0] * len(self.card_encoder.classes_)
            for move in state.get('table', []):
                if 'attack_card' in move:
                    card = move['attack_card']['card']
                    idx = self.card_encoder.transform([card])[0]
                    table_encoded[idx] = 1
                if 'defend_card' in move:
                    card = move['defend_card']['card']
                    idx = self.card_encoder.transform([card])[0]
                    table_encoded[idx] = 1
                    
            # 4. Текущий стейт
            state_encoded = [0] * 5
            state_mapping = {
                'attack': 0,
                'defend': 1, 
                'take': 2,
                'pass': 3,
                'bat': 4
            }
            current_state = state['players'][0]['state']
            state_encoded[state_mapping.get(current_state, 0)] = 1
            
            return np.concatenate([
                trump_encoded,
                hand_encoded,
                table_encoded,
                state_encoded
            ])
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def _determine_action(self, prev_state, current_state):
        """Определяет действие между двумя состояниями"""
        try:
            current_player_id = current_state['players'][0]['id']
            
            # 1. Проверяем изменения на столе (атака/защита)
            if len(current_state['table']) > len(prev_state['table']):
                last_move = current_state['table'][-1]
                if last_move['attack_card']['user_id'] == current_player_id:
                    return ('attack', last_move['attack_card']['card'])
                elif 'defend_card' in last_move and last_move['defend_card']['user_id'] == current_player_id:
                    return ('defend', last_move['defend_card']['card'])
            
            # 2. Проверяем изменения состояния (take/pass/bat)
            prev_player_state = prev_state['players'][0]['state']
            current_player_state = current_state['players'][0]['state']
            
            if current_player_state != prev_player_state:
                action_map = {
                    'take': ('take', None),
                    'pass': ('pass', None),
                    'bat': ('bat', None)
                }
                return action_map.get(current_player_state)
                
            return None
        except Exception as e:
            print(f"Action detection error: {str(e)}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Преобразуем действия в числовой формат
        action_type, action_card = sample['action']
        action_type_idx = self.action_encoder.transform([action_type])[0]
        action_card_idx = self.card_encoder.transform([action_card])[0] if action_card else 0
        
        return {
            'features': torch.FloatTensor(sample['features']),
            'action_type': torch.LongTensor([action_type_idx]),
            'action_card': torch.LongTensor([action_card_idx]),
            'is_winner': torch.FloatTensor([float(sample['is_winner'])]),
            'weight': torch.FloatTensor([1.5 if sample['is_winner'] else 1.0])
        }

    def get_sample_info(self, idx):
        """Возвращает читаемую информацию о семпле для отладки"""
        sample = self.samples[idx]
        game_state = sample['game_state']
        action_type, action_card = sample['action']
        
        return {
            'game_id': game_state.get('game_id', 'unknown'),
            'trump': game_state.get('trump', 'unknown'),
            'player_state': game_state['players'][0]['state'],
            'player_hand': game_state['players'][0]['hand'],
            'action': f"{action_type} {action_card if action_card else ''}",
            'is_winner_move': sample['is_winner'],
            'features_length': len(sample['features'])
        }