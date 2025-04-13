import asyncio
import json
import websockets
import random
from datetime import datetime

async def handle_connection(websocket, path):
    print("New client connected")
    
    # Отправляем сообщение о подключении
    connection_message = {
        "type": "connection_established",
        "player_id": "player_1"
    }
    print(f"Sending connection message: {connection_message}")
    await websocket.send(json.dumps(connection_message))
    
    # Эмуляция игры
    suits = ['S', 'C', 'D', 'H']
    ranks = ['9', '10', '11', '12', '13', '14']
    deck = [f"{rank}{suit}" for rank in ranks for suit in suits]
    random.shuffle(deck)
    
    # Начальное состояние игры
    state = {
        "game_id": "test_game_1",
        "trump": random.choice(deck),
        "deck": deck[:24],
        "bat": [],
        "table": [],
        "players": [
            {
                "id": "player_1",
                "state": "attack",
                "hand": deck[:6]
            },
            {
                "id": "player_2",
                "state": "defend",
                "hand": deck[6:12]
            }
        ],
        "game_rules": {
            "game_type": 0,
            "player_amount": 2,
            "card_amount": 24
        },
        "timestamp": datetime.now().isoformat()
    }
    
    turn = 0
    while turn < 5:
        # Отправляем текущее состояние игры
        game_state_message = {
            "type": "game_state",
            "state": state
        }
        print(f"Turn {turn + 1}: Sending game state: {game_state_message}")
        await websocket.send(json.dumps(game_state_message))
        
        # Получаем действие от клиента
        try:
            print("Waiting for action from client...")
            action = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            action_data = json.loads(action)
            print(f"Received action: {action_data}")
            
            # Эмулируем обработку действия
            if action_data["action"] == "attack":
                state["table"].append({
                    "attack_card": {
                        "card": action_data["card"],
                        "user_id": "player_1"
                    }
                })
                state["players"][0]["hand"].remove(action_data["card"])
                state["players"][0]["state"] = "wait"
                state["players"][1]["state"] = "defend"
            elif action_data["action"] == "defend":
                state["table"][-1]["defend_card"] = {
                    "card": action_data["card"],
                    "user_id": "player_1"
                }
                state["players"][0]["hand"].remove(action_data["card"])
                state["players"][0]["state"] = "wait"
                state["players"][1]["state"] = "attack"
            elif action_data["action"] in ["bat", "pass", "take"]:
                state["players"][0]["state"] = action_data["action"]
                state["players"][1]["state"] = "attack" if action_data["action"] != "take" else "wait"
                if action_data["action"] == "bat":
                    state["bat"].extend([pair["attack_card"]["card"] for pair in state["table"]])
                    state["table"] = []
            
            turn += 1
            state["timestamp"] = datetime.now().isoformat()
            
        except asyncio.TimeoutError:
            print("Client timed out")
            break
        except Exception as e:
            print(f"Error processing action: {e}")
            break
    
    # Отправляем сообщение о завершении игры
    game_over_message = {
        "type": "game_over",
        "winner": "player_1" if len(state["players"][0]["hand"]) == 0 else "player_2"
    }
    print(f"Sending game over message: {game_over_message}")
    await websocket.send(json.dumps(game_over_message))
    print("Game over, connection closed")

async def main():
    server = await websockets.serve(handle_connection, "localhost", 8000)
    print("WebSocket server started on ws://localhost:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())