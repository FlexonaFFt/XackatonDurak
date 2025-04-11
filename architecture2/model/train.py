import torch
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.utils.data import DataLoader
from utils.preprocessing import DurakDataset
from .model import DurakModel
from torch.optim import Adam
import torch.nn.functional as F

def train():
    # Инициализация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DurakDataset(hf_dataset="neuronetties/durak", split="train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = DurakModel(
        input_size=dataset[0]['features'].shape[0],
        action_classes=len(dataset.action_encoder.classes_),
        card_classes=len(dataset.card_encoder.classes_)
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Цикл обучения
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for batch in loader:
            features = batch['features'].to(device)
            action_type = batch['action_type'].to(device)
            action_card = batch['action_card'].to(device)
            weights = batch['weight'].to(device)
            is_winner = batch['is_winner'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            action_logits, card_logits, win_prob = model(features)
            
            # Расчет потерь с учетом весов
            action_loss = F.cross_entropy(
                action_logits.squeeze(1), 
                action_type.squeeze(1),
                weight=weights
            )
            
            card_loss = F.cross_entropy(
                card_logits.squeeze(1),
                action_card.squeeze(1),
                weight=weights
            )
            
            win_loss = F.binary_cross_entropy(
                win_prob.squeeze(1),
                is_winner.squeeze(1)
            )
            
            total_loss = action_loss + card_loss + 0.5 * win_loss
            total_loss.backward()
            optimizer.step()
            
            total_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    return model

if __name__ == '__main__':
    print("Train script started!")
    train()