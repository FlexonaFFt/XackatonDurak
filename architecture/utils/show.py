import pandas as pd
from datasets import load_dataset
import json

def display_dataset_samples(dataset_name="neuronetties/durak", num_samples=3):
    """
    Загружает и отображает первые несколько записей датасета
    с красивым форматированием JSON
    """
    try:
        dataset = load_dataset(dataset_name)
        print(f"Всего записей в датасете: {len(dataset['train'])}\n")
        samples = [dataset['train'][i] for i in range(num_samples)]
        df = pd.DataFrame(samples)
        def pretty_print(item):
            if isinstance(item, (dict, list)):
                return json.dumps(item, indent=2, ensure_ascii=False)
            return str(item)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.width', 1000)
        
        formatted_df = df.applymap(pretty_print)
        
        print("Первые записи датасета:")
        print(formatted_df.head(num_samples))
        
        print("\nСтруктура одной записи:")
        for key, value in samples[0].items():
            print(f"{key}: {type(value)}")
            
        print("\nПример содержимого первого игрока:")
        print(json.dumps(samples[0]['players'][0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        print("Проверьте:")
        print("1. Установлен ли пакет datasets (pip install datasets)")
        print("2. Доступен ли датасет neuronetties/durak")
        print("3. Авторизованы ли вы в Hugging Face (huggingface-cli login)")

if __name__ == "__main__":
    display_dataset_samples(num_samples=3)