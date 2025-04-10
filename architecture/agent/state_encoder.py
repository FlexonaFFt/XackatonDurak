import numpy as np

# Константы
RANKS = ['9', '10', '11', '12', '13', '14']
SUITS = ['S', 'C', 'D', 'H']  # Пики, Трефы, Бубны, Червы

class StateEncoder:
    def __init__(self):
        self.rank_to_idx = {rank: i for i, rank in enumerate(RANKS)}
        self.suit_to_idx = {suit: i for i, suit in enumerate(SUITS)}
    
    def encode_card(self, card):
        """Кодирует карту в вектор [номинал, масть]."""
        if card is None:
            return [0, 0]  # Пустая карта
        rank = card[:-1]
        suit = card[-1]
        return [self.rank_to_idx[rank], self.suit_to_idx[suit]]
    
    def encode_game_state(self, game_state, player_id):
        """Преобразует состояние игры в числовой вектор."""
        # Кодирование руки игрока (максимум 6 карт)
        hand = game_state['players'][player_id]['hand']
        hand_encoded = np.zeros((6, 2))
        for i, card in enumerate(hand):
            hand_encoded[i] = self.encode_card(card)
        
        # Кодирование стола (максимум 6 атак + 6 защит)
        table_encoded = np.zeros((12, 2))
        for i, card_info in enumerate(game_state['table']):
            table_encoded[2*i] = self.encode_card(card_info['attack_card']['card'])
            if 'defend_card' in card_info:
                table_encoded[2*i + 1] = self.encode_card(card_info['defend_card']['card'])
        
        # Кодирование козыря и типа игры
        trump_encoded = self.suit_to_idx[game_state['trump'][-1]]
        game_type = game_state['game_rules']['game_type']
        
        # Объединение всех признаков
        state_vector = np.concatenate([
            hand_encoded.flatten(),
            table_encoded.flatten(),
            [trump_encoded, game_type]
        ])
        return state_vector