import torch 

def predict_move(model, game_state, card_encoder, action_encoder):
    # Преобразование состояния в тензор
    features = torch.FloatTensor(dataset._extract_features(game_state)).unsqueeze(0)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        action_logits, card_logits = model(features)
        
        action_pred = torch.argmax(action_logits, dim=1).item()
        card_pred = torch.argmax(card_logits, dim=1).item()
        
        action_type = action_encoder.inverse_transform([action_pred])[0]
        card = card_encoder.inverse_transform([card_pred])[0] if card_pred != card_encoder.transform(['None'])[0] else None
    
    # Формирование ответа
    if action_type in ['attack', 'defend'] and card:
        return {'type': action_type, 'move': card}
    else:
        return {'type': 'state', 'state': action_type}