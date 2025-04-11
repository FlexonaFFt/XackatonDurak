import torch
import torch.nn as nn

class DurakModel(nn.Module):
    def __init__(self, input_size, action_classes, card_classes):
        super(DurakModel, self).__init__()
        
        # Общие слои
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Голова для типа действия
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_classes)
        )
        
        # Голова для выбора карты
        self.card_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, card_classes)
        )
        
        # Голова для предсказания "победности" хода
        self.win_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.action_head(shared)
        card_logits = self.card_head(shared)
        win_prob = self.win_head(shared)
        return action_logits, card_logits, win_prob