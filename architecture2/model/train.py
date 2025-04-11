from model import DurakModel
from utils.preprocessing import DataLoader
from utils.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam

import torch 
import torch.nn as nn
import torch.optim as optim

def train_model():
    # Инициализация датасета
    dataset = DurakDataset('path_to_data')
    
    # Разделение на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Инициализация модели
    input_size = dataset[0][0].shape[0]
    model = DurakModel(
        input_size=input_size,
        action_classes=dataset.action_classes,
        card_classes=dataset.card_classes
    )
    
    # Оптимизатор и функция потерь
    optimizer = Adam(model.parameters(), lr=0.001)
    action_loss_fn = nn.CrossEntropyLoss()
    card_loss_fn = nn.CrossEntropyLoss()
    
    # Цикл обучения
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        
        for X, (y_action, y_card) in train_loader:
            optimizer.zero_grad()
            
            action_logits, card_logits = model(X)
            
            # Вычисление потерь
            action_loss = action_loss_fn(action_logits, y_action)
            card_loss = card_loss_fn(card_logits, y_card)
            total_loss = action_loss + card_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct_actions = 0
        correct_cards = 0
        total = 0
        
        with torch.no_grad():
            for X, (y_action, y_card) in val_loader:
                action_logits, card_logits = model(X)
                
                action_loss = action_loss_fn(action_logits, y_action)
                card_loss = card_loss_fn(card_logits, y_card)
                val_loss += (action_loss + card_loss).item()
                
                # Точность предсказаний
                _, action_preds = torch.max(action_logits, 1)
                _, card_preds = torch.max(card_logits, 1)
                
                correct_actions += (action_preds == y_action).sum().item()
                correct_cards += (card_preds == y_card).sum().item()
                total += y_action.size(0)
        
        print(f'Epoch {epoch+1}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Action Accuracy: {correct_actions/total:.4f}')
        print(f'Card Accuracy: {correct_cards/total:.4f}\n')
    
    return model, dataset.card_encoder, dataset.action_encoder