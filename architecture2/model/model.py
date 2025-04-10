import torch.nn as nn
import torch.nn.functional as F

class DurakModel(nn.Module):
    def __init__(self, input_size, action_classes, card_classes):
        super(DurakModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(64, action_classes)
        self.card_head = nn.Linear(64, card_classes)
        
    def forward(self, x):
        shared = self.shared_layers(x)
        action_logits = self.action_head(shared)
        card_logits = self.card_head(shared)
        return action_logits, card_logits