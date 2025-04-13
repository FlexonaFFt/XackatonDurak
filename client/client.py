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
        print(f"Sending message: {message}")
        await websocket.send(json.dumps(message))

    async def receive(self, websocket):
        print("Waiting for message from server...")
        message = await websocket.recv()
        data = json.loads(message)
        print(f"Received: {data}")
        return data

    async def connect(self):
        print(f"Attempting to connect to {self.uri}...")
        try:
            async with websockets.connect(self.uri) as websocket:
                print("Connected to server.")
                
                # Получаем начальное сообщение
                data = await self.receive(websocket)
                
                if data['type'] == 'connection_established':
                    self.player_id = data['player_id']
                    print(f"Player ID: {self.player_id}")
                else:
                    print(f"Unexpected message type: {data['type']}")
                    return
                
                while True:
                    data = await self.receive(websocket)
                    
                    if data['type'] == 'game_state':
                        # Отправляем состояние игры модели
                        model_request = {
                            "snapshot": data['state'],
                            "my_id": self.player_id
                        }
                        
                        try:
                            print("Sending request to model...")
                            response = requests.post(self.model_endpoint, json=model_request)
                            response.raise_for_status()
                            action = response.json()
                            print(f"Model predicted action: {action}")
                        except Exception as e:
                            print(f"Error getting prediction from model: {e}")
                            action = {"action": "pass"}
                        
                        # Отправляем действие на сервер
                        await self.send(websocket, action)
                    
                    elif data['type'] == 'game_over':
                        print("Game over!")
                        print(f"Winner: {data['winner']}")
                        break
                    
                    elif data['type'] == 'error':
                        print(f"Error: {data['message']}")
                        break
        except Exception as e:
            print(f"Connection error: {e}")

async def main():
    server_uri = "ws://localhost:8000"
    model_endpoint = "http://localhost:5000"
    
    client = DurakClient(server_uri, model_endpoint)
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main())