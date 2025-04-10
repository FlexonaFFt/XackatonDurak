import torch
import torch.nn as nn
import torch.nn.functional as F

class CardEmbedding(nn.Module):
    """Слой для embedding карт (номинал + масть)"""
    def __init__(self):
        super().__init__()
        self.rank_emb = nn.Embedding(6, 8)  # 6 номиналов (9-14)
        self.suit_emb = nn.Embedding(4, 4)   # 4 масти
        
    def forward(self, cards):
        # cards: [batch_size, seq_len] tensor закодированных карт
        ranks = (cards // 4).long()  # Номинал 9-14
        suits = (cards % 4).long()   # Масть
        return torch.cat([
            self.rank_emb(ranks),
            self.suit_emb(suits)
        ], dim=-1)

class DurakModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        
        # Embedding компоненты
        self.card_emb = CardEmbedding()
        self.state_emb = nn.Embedding(7, 8)  # 7 возможных состояний игрока
        
        # Основные слои обработки
        self.card_lstm = nn.LSTM(12, hidden_size//2, bidirectional=True)  # 12 = 8(номинал) + 4(масть)
        
        # Attention механизм для анализа стола
        self.table_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Политика и Value головки
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64)  # 64 возможных действия
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x - словарь с входными данными:
        #   hand: карты в руке [batch, max_hand_size]
        #   table: карты на столе [batch, max_table_size]
        #   trump: козырь [batch]
        #   player_state: состояние игрока [batch]
        #   deck_size: размер колоды [batch]
        
        # Embedding карт в руке
        hand_emb = self.card_emb(x['hand'])
        hand_features, _ = self.card_lstm(hand_emb)
        
        # Обработка карт на столе
        table_emb = self.card_emb(x['table'])
        table_features, _ = self.card_lstm(table_emb)
        
        # Attention между картами в руке и на столе
        attn_out, _ = self.table_attention(
            hand_features, table_features, table_features
        )
        
        # Объединение признаков
        state_emb = self.state_emb(x['player_state'])
        trump_emb = self.card_emb(x['trump'].unsqueeze(1)).squeeze(1)
        
        global_features = torch.cat([
            hand_features.mean(dim=1),
            attn_out.mean(dim=1),
            state_emb,
            trump_emb,
            x['deck_size'].unsqueeze(1).float()
        ], dim=1)
        
        # Выходы модели
        policy_logits = self.policy_head(global_features)
        value = self.value_head(global_features)
        
        return policy_logits, value