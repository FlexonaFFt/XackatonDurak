import torch
import torch.nn as nn
import numpy as np
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

# Модель нейросети (такая же, как в вашем коде)
class DurakNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_cards=24, num_actions=4):
        super(DurakNet, self).__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)
        self.card_head = nn.Linear(hidden_dim // 2, num_cards)
        self.state_head = nn.Linear(hidden_dim // 2, 3)
        
    def forward(self, state):
        encoded = self.state_encoder(state)
        action_logits = self.action_head(encoded)
        card_logits = self.card_head(encoded)
        state_logits = self.state_head(encoded)
        return action_logits, card_logits, state_logits

# Кодирование состояния (без изменений)
def encode_state(snapshot, my_id, device='cpu'):
    try:
        data = snapshot
        
        state = np.zeros(128, dtype=np.float32)
        
        trump = data['trump']
        suits = {'S': 0, 'C': 1, 'D': 2, 'H': 3}
        ranks = {'9': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5}
        trump_rank, trump_suit = trump[:-1], trump[-1]
        state[0 + suits[trump_suit]] = 1.0
        state[4 + ranks[trump_rank]] = 1.0
        
        for player in data['players']:
            if player['id'] == my_id:
                for card in player['hand']:
                    rank, suit = card[:-1], card[-1]
                    card_idx = ranks[rank] * 4 + suits[suit]
                    state[10 + card_idx] = 1.0
        
        for i, pair in enumerate(data['table'][:4]):
            attack_card = pair['attack_card']['card']
            rank, suit = attack_card[:-1], attack_card[-1]
            state[34 + i * 12 + ranks[rank]] = 1.0
            state[34 + i * 12 + 6 + suits[suit]] = 1.0
            if 'defend_card' in pair:
                defend_card = pair['defend_card']['card']
                rank, suit = defend_card[:-1], defend_card[-1]
                state[82 + i * 12 + ranks[rank]] = 1.0
                state[82 + i * 12 + 6 + suits[suit]] = 1.0
        
        state[114] = len(data['deck']) / 24.0
        
        for player in data['players']:
            if player['id'] == my_id:
                state_map = {'attack': 115, 'defend': 116, 'bat': 117, 'pass': 118, 'take': 119}
                if player['state'] in state_map:
                    state[state_map[player['state']]] = 1.0
        
        state[120] = data['game_rules']['game_type']
        
        tensor = torch.tensor(state, dtype=torch.float32, device=device, requires_grad=False)
        return tensor
    
    except Exception as e:
        print(f"Error in encode_state: {e}")
        return torch.zeros(128, dtype=torch.float32, device=device)

# Предсказание действия (адаптировано для durak_client)
def predict_action(model, snapshot, my_id, device='cpu'):
    model.eval()
    with torch.no_grad():
        state = encode_state(snapshot, my_id, device=device)
        action_logits, card_logits, state_logits = model(state.unsqueeze(0))
        
        action_idx = torch.argmax(action_logits, dim=1).item()
        action_map = {0: 'attack', 1: 'defend', 2: 'state', 3: 'wait'}
        action_type = action_map[action_idx]
        
        if action_type == 'attack' or action_type == 'defend':
            card_idx = torch.argmax(card_logits, dim=1).item()
            ranks = {0: '9', 1: '10', 2: '11', 3: '12', 4: '13', 5: '14'}
            suits = {0: 'S', 1: 'C', 2: 'D', 3: 'H'}
            card = f"{ranks[card_idx // 4]}{suits[card_idx % 4]}"
            # Формат для durak_client
            return {"action": action_type, "card": card}
        elif action_type == 'state':
            state_idx = torch.argmax(state_logits, dim=1).item()
            state_map = {0: 'bat', 1: 'pass', 2: 'take'}
            # durak_client ожидает конкретное действие
            action = state_map[state_idx]
            return {"action": action}
        else:
            return {"action": "pass"}

# Загрузка модели
def load_model(model_path="durak_model.pth", device='cpu'):
    model = DurakNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model weights loaded from {model_path}")
        return model
    else:
        print(f"Model file {model_path} not found")
        return None

# HTTP-сервер для взаимодействия с durak_client
class ModelRequestHandler(BaseHTTPRequestHandler):
    model = None
    device = None
    
    @classmethod
    def set_model(cls, model, device):
        cls.model = model
        cls.device = device

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            snapshot = request_data['snapshot']
            my_id = request_data['my_id']
            
            # Предсказание действия
            action = predict_action(self.model, snapshot, my_id, self.device)
            
            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(action).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

def run_model_service(model, device, host='localhost', port=5000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, ModelRequestHandler)
    ModelRequestHandler.set_model(model, device)
    print(f"Starting model service on {host}:{port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Загружаем модель
    model = load_model("durak_model.pth", device=device)
    if model is None:
        print("Exiting: Model not loaded.")
        exit(1)
    
    # Запускаем сервер
    run_model_service(model, device)