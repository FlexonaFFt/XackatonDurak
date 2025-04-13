import json
import sys
import time
import asyncio
import requests
import websockets
from typing import Dict

class DurakClient:
    def __init__(self, uri: str, model_endpoint: str):
        self.uri = uri
        self.model_endpoint = model_endpoint
        self.player_id = None

    async def send(self, websocket, message: Dict):
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")

    async def receive(self, websocket):
        message = await websocket.recv()
        data = json.loads(message)
        print(f"Received: {data}")
        return data

    async def connect(self):
        async with websockets.connect(self.uri) as websocket:
            print("Connected to server.")
            
            # Получаем начальное сообщение
            data = await self.receive(websocket)
            
            if data['type'] == 'connection_established':
                self.player_id = data['player_id']
                print(f"Player ID: {self.player_id}")
            
            while True:
                data = await self.receive(websocket)
                
                if data['type'] == 'game_state':
                    # Отправляем состояние игры модели
                    model_request = {
                        "snapshot": data['state'],
                        "my_id": self.player_id
                    }
                    
                    try:
                        # Запрашиваем предсказание у модели
                        response = requests.post(self.model_endpoint, json=model_request)
                        response.raise_for_status()
                        action = response.json()
                        print(f"Model predicted action: {action}")
                    except Exception as e:
                        print(f"Error getting prediction from model: {e}")
                        action = {"action": "pass"}  # Запасной вариант
                    
                    # Отправляем действие на сервер
                    await self.send(websocket, action)
                
                elif data['type'] == 'game_over':
                    print("Game over!")
                    print(f"Winner: {data['winner']}")
                    break
                
                elif data['type'] == 'error':
                    print(f"Error: {data['message']}")
                    break

async def main():
    # URI сервера (можно изменить, если тестируете локально)
    server_uri = "wss://durak-game.varkus.ru:8000"
    # URI вашего локального сервиса модели
    model_endpoint = "http://localhost:5000"
    
    client = DurakClient(server_uri, model_endpoint)
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main())