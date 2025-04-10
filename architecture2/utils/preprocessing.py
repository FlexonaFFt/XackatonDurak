import os
import torch
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class DurakDataset(Dataset):
    def __init__(self, data_dir):
        self.card_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self._init_encoders()
        
        self.X, self.y_action, self.y_card = self._load_data(data_dir)
        
    def _init_encoders(self):
        cards = [f"{value}{suit}" for value in ['9','10','11','12','13','14'] 
                               for suit in ['S','C','D','H']]
        self.card_encoder.fit(cards)
        
        actions = ['attack', 'defend', 'take', 'pass', 'bat']
        self.action_encoder.fit(actions)
        self.card_classes = len(self.card_encoder.classes_)
        self.action_classes = len(self.action_encoder.classes_)

    def _load_data(self, data_dir):
        matches = defaultdict(list)
        
        # Загрузка и группировка данных
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename)) as f:
                    data = json.load(f)
                    matches[data['game_id']].append(data)
        
        all_X, all_y_action, all_y_card = [], [], []
        
        for match_id, states in matches.items():
            states_sorted = sorted(states, key=lambda x: x['timestamp'])
            
            for i in range(1, len(states_sorted)):
                prev_state = states_sorted[i-1]
                current_state = states_sorted[i]
                
                features = self._extract_features(prev_state)
                action = self._extract_action(prev_state, current_state)
                
                if action:
                    all_X.append(features)
                    all_y_action.append(action[0])
                    all_y_card.append(action[1] if action[1] else 'None')
        
        # Преобразование в тензоры
        X_tensor = torch.FloatTensor(np.array(all_X))
        y_action_tensor = torch.LongTensor(self.action_encoder.transform(all_y_action))
        y_card_tensor = torch.LongTensor(self.card_encoder.transform(all_y_card))
        
        return X_tensor, y_action_tensor, y_card_tensor

    def _extract_features(self, state):
        # Аналогично предыдущей реализации, но возвращает numpy array
        trump_suit = state['trump'][-1]
        trump_encoded = [0, 0, 0, 0]
        trump_encoded[['S','C','D','H'].index(trump_suit)] = 1
        
        hand_encoded = [0] * self.card_classes
        for card in state['players'][0]['hand']:
            idx = self.card_encoder.transform([card])[0]
            hand_encoded[idx] = 1
            
        table_encoded = [0] * self.card_classes
        for move in state['table']:
            attack_card = move['attack_card']['card']
            idx = self.card_encoder.transform([attack_card])[0]
            table_encoded[idx] = 1
            
            if 'defend_card' in move:
                defend_card = move['defend_card']['card']
                idx = self.card_encoder.transform([defend_card])[0]
                table_encoded[idx] = 1
                
        state_encoded = [0]*5
        player_state = state['players'][0]['state']
        if player_state == 'attack': state_encoded[0] = 1
        elif player_state == 'defend': state_encoded[1] = 1
        elif player_state == 'take': state_encoded[2] = 1
        elif player_state == 'pass': state_encoded[3] = 1
        elif player_state == 'bat': state_encoded[4] = 1
        
        return np.array(trump_encoded + hand_encoded + table_encoded + state_encoded)

    def _extract_action(self, prev_state, current_state):
        # Аналогично предыдущей реализации
        current_player_id = prev_state['players'][0]['id']
        
        if len(current_state['table']) > len(prev_state['table']):
            new_move = current_state['table'][-1]
            if new_move['attack_card']['user_id'] == current_player_id:
                return ('attack', new_move['attack_card']['card'])
            elif 'defend_card' in new_move and new_move['defend_card']['user_id'] == current_player_id:
                return ('defend', new_move['defend_card']['card'])
        elif current_state['players'][0]['state'] == 'take' and prev_state['players'][0]['state'] != 'take':
            return ('take', None)
        elif current_state['players'][0]['state'] == 'pass' and prev_state['players'][0]['state'] != 'pass':
            return ('pass', None)
        elif current_state['players'][0]['state'] == 'bat' and prev_state['players'][0]['state'] != 'bat':
            return ('bat', None)
        return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], (self.y_action[idx], self.y_card[idx])