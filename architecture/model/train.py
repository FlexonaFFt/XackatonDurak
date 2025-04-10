from model import DurakModel

import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(dataloader, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DurakModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            policy_targets = batch['policy_target'].to(device)
            value_targets = batch['value_target'].to(device)
            
            # Forward pass
            policy_logits, value = model(inputs)
            
            # Расчет потерь
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            value_loss = F.mse_loss(value.squeeze(), value_targets)
            loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()