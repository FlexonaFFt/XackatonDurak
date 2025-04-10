import torch
import numpy as np

from agent.logic import DurakGame
from agent.state_encoder import StateEncoder
from model import DQN, Agent
from config import Config

def initialize_game():
    """Инициализирует игру с двумя игроками."""
    game = DurakGame(game_type=0)  # 0 - классический дурак
    player_ids = ["player1", "player2"]
    game.start_game(player_ids)
    return game

def train_dqn():
    # Инициализация
    encoder = StateEncoder()
    game = initialize_game()
    state_size = 6*2 + 12*2 + 2  # Размер вектора состояния
    action_size = 50  # Примерное количество возможных действий
    agent = Agent(state_size, action_size)
    
    # Гиперпараметры
    batch_size = Config.BATCH_SIZE
    episodes = Config.EPISODES
    
    for episode in range(episodes):
        game = initialize_game()
        state = encoder.encode_game_state(game.get_game_state(), "player1")
        total_reward = 0
        
        while not game.game_over:
            # Ход агента
            action_idx = agent.act(state)
            action = _map_action_idx_to_move(action_idx, game, "player1")
            
            # Применение хода
            game.make_move("player1", action)
            next_state = encoder.encode_game_state(game.get_game_state(), "player1")
            
            # Награда
            reward = _calculate_reward(game, "player1")
            done = game.game_over
            
            # Сохранение опыта
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Обучение на мини-батче
            agent.replay(batch_size)
        
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    # Сохранение модели
    torch.save(agent.model.state_dict(), "models/saved_models/dqn_durak.pth")

def _map_action_idx_to_move(action_idx, game, player_id):
    """Преобразует индекс действия в конкретный ход."""
    # Пример: action_idx 0-5 - карты в руке, 6 - 'pass', 7 - 'bat', 8 - 'take'
    player = next(p for p in game.players if p['id'] == player_id)
    if action_idx < len(player['hand']):
        return {"type": "attack", "move": player['hand'][action_idx]}
    elif action_idx == 6:
        return {"type": "state", "state": "pass"}
    elif action_idx == 7:
        return {"type": "state", "state": "bat"}
    elif action_idx == 8:
        return {"type": "state", "state": "take"}
    else:
        return {"type": "wait"}

def _calculate_reward(game, player_id):
    """Вычисляет награду для игрока."""
    if game.game_over:
        return 1.0 if game.winner == player_id else -1.0
    # Промежуточные награды
    return 0.01  # Например, за каждый успешный ход

if __name__ == "__main__":
    train_dqn()