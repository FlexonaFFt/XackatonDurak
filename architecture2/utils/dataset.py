from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DurakDataset(Dataset):
    def __init__(self, split="train", limit=None):
        """
        Args:
            split: "train" или "validation"
            limit: ограничение количества примеров
        """
        # Загружаем датасет напрямую без преобразования
        self.data = list(load_dataset("neuronetties/durak", split=split).select(range(limit)) if limit else list(load_dataset("neuronetties/durak", split=split)))
        
        # Инициализация кодировщиков
        self.card_encoder = LabelEncoder()
        cards = [f"{value}{suit}" for value in ['9','10','11','12','13','14'] 
                               for suit in ['S','C','D','H']]
        self.card_encoder.fit(cards + ['None'])
        
        self.action_encoder = LabelEncoder()
        actions = ['attack', 'defend', 'take', 'pass', 'bat']
        self.action_encoder.fit(actions)

    def _extract_features(self, game_state):
        """Преобразование состояния игры в вектор признаков"""
        try:
            # 1. Козырь
            trump = game_state.get('trump', '14S')  # Значение по умолчанию
            trump_suit = trump[-1]
            trump_encoded = [0, 0, 0, 0]
            suits = ['S', 'C', 'D', 'H']
            trump_encoded[suits.index(trump_suit)] = 1
            
            # 2. Карты в руке (первый игрок)
            hand_encoded = [0] * len(self.card_encoder.classes_)
            for card in game_state.get('players', [{}])[0].get('hand', []):
                idx = self.card_encoder.transform([card])[0]
                hand_encoded[idx] = 1
                
            # 3. Карты на столе
            table_encoded = [0] * len(self.card_encoder.classes_)
            for move in game_state.get('table', []):
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
            
            # 4. Текущий стейт игрока
            state_encoded = [0, 0, 0, 0, 0]
            player_state = game_state.get('players', [{}])[0].get('state', '')
            if player_state == 'attack': state_encoded[0] = 1
            elif player_state == 'defend': state_encoded[1] = 1
            elif player_state == 'take': state_encoded[2] = 1
            elif player_state == 'pass': state_encoded[3] = 1
            elif player_state == 'bat': state_encoded[4] = 1
            
            return np.concatenate([trump_encoded, hand_encoded, table_encoded, state_encoded])
        
        except Exception as e:
            print(f"Error processing game state: {e}")
            return np.zeros(4 + len(self.card_encoder.classes_) * 2 + 5)  # Возвращаем нулевой вектор при ошибке

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_state = self.data[idx]
        features = self._extract_features(game_state)
        
        # Заглушка для действий
        action = 0
        card = 0
        
        return (
            torch.FloatTensor(features),
            torch.LongTensor([action]),
            torch.LongTensor([card])
        )