from dataset import DurakDataset
import json

def print_sample_info(sample_info):
    """Красиво выводит информацию о семпле"""
    print("\n=== Sample Info ===")
    print(f"Game ID: {sample_info['game_id']}")
    print(f"Trump: {sample_info['trump']}")
    print(f"Player state: {sample_info['player_state']}")
    print(f"Player hand: {sample_info['player_hand']}")
    print(f"Action: {sample_info['action']}")
    print(f"Is winner move: {sample_info['is_winner_move']}")
    print(f"Features vector length: {sample_info['features_length']}")

if __name__ == "__main__":
    print("Testing dataset loading...")
    
    try:
        # Загружаем только 10 игр для теста
        dataset = DurakDataset(limit=100)
        
        print("\nDataset summary:")
        print(f"Total samples: {len(dataset)}")
        print(f"Card classes: {len(dataset.card_encoder.classes_)}")
        print(f"Action classes: {list(dataset.action_encoder.classes_)}")
        
        # Проверяем первый и случайный семплы
        print_sample_info(dataset.get_sample_info(0))
        print_sample_info(dataset.get_sample_info(len(dataset)//2))
        
        # Проверяем структуру сырых данных
        print("\nRaw data structure example:")
        print(json.dumps(dataset.data[0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your dataset structure.")