import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import defaultdict

class DurakDataset(Dataset):
    def __init__(self, data_path=None, hf_dataset="neuronetties/durak", split="train", limit=None):
        """
        Args:
            data_path: путь к локальным данным (если не используется HF)
            hf_dataset: имя датасета на HuggingFace
            split: "train" или "validation"
            limit: ограничение количества примеров
        """
        self.card_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self._init_encoders()
        
        # Загрузка данных
        if data_path:
            self.data = self._load_from_files(data_path, limit)
        else:
            from datasets import load_dataset
            hf_data = load_dataset(hf_dataset, split=split)
            self.data = list(hf_data.select(range(limit))) if limit else list(hf_data)
        
        print(f"Loaded {len(self.data)} raw samples")
        self.games = self._group_by_games()
        print(f"Grouped into {len(self.games)} games")
        self.samples = self._prepare_samples()
        print(f"Prepared {len(self.samples)} training samples")

        if len(self.samples) == 0:
            raise ValueError("No valid training samples found! Check your dataset structure.")

    def _init_encoders(self):
        cards = [f"{value}{suit}" for value in ['9','10','11','12','13','14'] 
                               for suit in ['S','C','D','H']]
        self.card_encoder.fit(cards + ['None'])
        
        actions = ['attack', 'defend', 'take', 'pass', 'bat']
        self.action_encoder.fit(actions)

    def _group_by_games(self):
        """Группирует состояния по game_id"""
        games = defaultdict(list)
        for state in self.data:
            games[state['game_id']].append(state)
        return games

    def _prepare_samples(self):
        """Создает обучающие примеры, ориентируясь на победителей"""
        samples = []
        
        for game_id, states in self.games.items():
            try:
                # Сортируем по timestamp (с проверкой наличия ключа)
                sorted_states = sorted(
                    [s for s in states if 'timestamp' in s],
                    key=lambda x: x['timestamp']
                )
                
                if not sorted_states:
                    continue
                    
                # Определяем победителя (с проверкой наличия ключа)
                winner_id = sorted_states[-1].get('winner')
                if not winner_id:
                    continue
                    
                for i in range(1, len(sorted_states)):
                    prev_state = sorted_states[i-1]
                    current_state = sorted_states[i]
                    
                    # Проверяем наличие необходимых полей
                    if not all(key in prev_state for key in ['players', 'table', 'trump']):
                        continue
                    if not all(key in current_state for key in ['players']):
                        continue
                    
                    # Определяем, кто сделал ход
                    current_player_id = current_state['players'][0].get('id')
                    if not current_player_id:
                        continue
                        
                    is_winner_move = (current_player_id == winner_id)
                    
                    features = self._extract_features(prev_state)
                    action = self._extract_action(prev_state, current_state)
                    
                    if action:
                        samples.append({
                            'features': features,
                            'action': action,
                            'is_winner_move': is_winner_move,
                            'game_state': prev_state
                        })
            
            except Exception as e:
                print(f"Error processing game {game_id}: {str(e)}")
                continue
        
        return samples

    def _extract_features(self, state):
        """Улучшенная версия с обработкой ошибок"""
        try:
            # 1. Козырь и текущий игрок
            trump = state.get('trump', '14S')
            trump_suit = trump[-1]
            trump_encoded = [0, 0, 0, 0]
            suits = ['S', 'C', 'D', 'H']
            trump_encoded[suits.index(trump_suit)] = 1
            
            # 2. Карты в руке (всех игроков)
            players_encoded = []
            for player in state.get('players', []):
                hand_encoded = [0] * len(self.card_encoder.classes_)
                for card in player.get('hand', []):
                    idx = self.card_encoder.transform([card])[0]
                    hand_encoded[idx] = 1
                players_encoded.extend(hand_encoded)
            
            # 3. Карты на столе
            table_encoded = [0] * len(self.card_encoder.classes_)
            for move in state.get('table', []):
                if 'attack_card' in move:
                    attack_card = move['attack_card'].get('card', '')
                    if attack_card:
                        idx = self.card_encoder.transform([attack_card])[0]
                        table_encoded[idx] = 1
                if 'defend_card' in move:
                    defend_card = move['defend_card'].get('card', '')
                    if defend_card:
                        idx = self.card_encoder.transform([defend_card])[0]
                        table_encoded[idx] = 1
            
            # 4. Текущий стейт
            state_encoded = [0]*5
            player_state = state.get('players', [{}])[0].get('state', '')
            if player_state == 'attack': state_encoded[0] = 1
            elif player_state == 'defend': state_encoded[1] = 1
            elif player_state == 'take': state_encoded[2] = 1
            elif player_state == 'pass': state_encoded[3] = 1
            elif player_state == 'bat': state_encoded[4] = 1
            
            return np.concatenate([
                trump_encoded,
                players_encoded,
                table_encoded,
                state_encoded
            ])
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(4 + 2*len(self.card_encoder.classes_) + len(self.card_encoder.classes_) + 5)

    def _extract_action(self, prev_state, current_state):
        """Определяет действие между состояниями с улучшенной логикой"""
        try:
            current_player_id = current_state['players'][0]['id']
            
            # Определение типа действия
            if len(current_state['table']) > len(prev_state['table']):
                new_move = current_state['table'][-1]
                if new_move['attack_card']['user_id'] == current_player_id:
                    return ('attack', new_move['attack_card']['card'])
                elif 'defend_card' in new_move and new_move['defend_card']['user_id'] == current_player_id:
                    return ('defend', new_move['defend_card']['card'])
            
            state_changes = {
                'take': ('take', None),
                'pass': ('pass', None),
                'bat': ('bat', None)
            }
            
            for new_state, action in state_changes.items():
                if (current_state['players'][0]['state'] == new_state and 
                    prev_state['players'][0]['state'] != new_state):
                    return action
                    
            return None
        except Exception as e:
            print(f"Action extraction error: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features']
        action_type, action_card = sample['action']
        
        # Преобразуем действия в числовой формат
        action_type_encoded = self.action_encoder.transform([action_type])[0]
        action_card_encoded = self.card_encoder.transform([action_card])[0] if action_card else 0
        
        # Вес примера (увеличиваем вес ходов победителя)
        weight = 1.5 if sample['is_winner_move'] else 1.0
        
        return {
            'features': torch.FloatTensor(features),
            'action_type': torch.LongTensor([action_type_encoded]),
            'action_card': torch.LongTensor([action_card_encoded]),
            'weight': torch.FloatTensor([weight]),
            'is_winner': torch.FloatTensor([1.0 if sample['is_winner_move'] else 0.0])
        }