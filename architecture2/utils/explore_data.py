from dataset import DurakDataset
import json

if __name__ == "__main__":
    print("Testing dataset loading...")
    
    try:
        # Загрузка с ограничением в 2 примера для теста
        ds = DurakDataset(limit=2)
        print(f"Total samples: {len(ds)}")
        
        # Проверка структуры первого элемента
        print("\nFirst sample structure:")
        print(json.dumps(ds.data[0], indent=2, ensure_ascii=False))
        
        # Проверка обработки признаков
        features, action, card = ds[0]
        print("\nProcessed features:")
        print(f"Shape: {features.shape}")
        print(f"Sample values: {features[:10]}...")  # Первые 10 значений
        
        # Проверка кодировщиков
        print("\nEncoders info:")
        print(f"Card classes: {len(ds.card_encoder.classes_)}")
        print(f"Action classes: {ds.action_encoder.classes_}")
        
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("Please check the dataset structure and preprocessing code.")