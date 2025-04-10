import pytest
from logic import DurakGame

@pytest.fixture
def classic_game():
    """Фикстура для классической игры"""
    game = DurakGame(game_type=0)
    game.start_game(['player1', 'player2'])
    return game

@pytest.fixture
def transfer_game():
    """Фикстура для переводного дурака"""
    game = DurakGame(game_type=1)
    game.start_game(['player1', 'player2'])
    return game

def test_game_initialization(classic_game):
    """Тест инициализации игры"""
    state = classic_game.get_game_state()
    
    assert len(state['deck']) == 24 - 12  # 24 карты минус 6 на игрока × 2
    assert len(state['players'][0]['hand']) == 6
    assert len(state['players'][1]['hand']) == 6
    assert state['trump'] in ['S', 'C', 'D', 'H']
    assert state['winner'] is None
    assert classic_game.game_over is False

def test_first_attack(classic_game):
    """Тест первой атаки"""
    attacker_id = classic_game.current_attacker
    defender_id = classic_game.current_defender
    
    # Получаем карту атакующего
    attacker = next(p for p in classic_game.players if p['id'] == attacker_id)
    attack_card = attacker['hand'][0]
    
    # Атакуем
    assert classic_game.make_move(attacker_id, {'type': 'attack', 'move': attack_card}) is True
    
    state = classic_game.get_game_state()
    assert len(state['table']) == 1
    assert state['table'][0]['attack_card']['card'] == attack_card
    assert attack_card not in attacker['hand']

def test_valid_defense(classic_game):
    """Тест валидной защиты"""
    # Подготовим тестовые данные
    classic_game.trump = 'H'  # Для предсказуемости теста
    attacker_id = classic_game.current_attacker
    defender_id = classic_game.current_defender
    
    # Дадим игрокам конкретные карты
    classic_game.players[0]['hand'] = ['10H']  # Атакующий
    classic_game.players[1]['hand'] = ['12H']  # Защитник
    
    # Атака
    classic_game.make_move(attacker_id, {'type': 'attack', 'move': '10H'})
    
    # Защита
    assert classic_game.make_move(
        defender_id, 
        {'type': 'defend', 'move': [[attacker_id, '10H'], [defender_id, '12H']]}
    ) is True
    
    state = classic_game.get_game_state()
    assert len(state['table']) == 1
    assert 'defend_card' in state['table'][0]
    assert state['table'][0]['defend_card']['card'] == '12H'
    assert '12H' not in classic_game.players[1]['hand']

def test_invalid_defense(classic_game):
    """Тест невалидной защиты"""
    classic_game.trump = 'H'
    attacker_id = classic_game.current_attacker
    defender_id = classic_game.current_defender
    
    classic_game.players[0]['hand'] = ['10S']  # Атакующий
    classic_game.players[1]['hand'] = ['9H']   # Защитник (неправильная защита)
    
    classic_game.make_move(attacker_id, {'type': 'attack', 'move': '10S'})
    
    # Попытка защиты младшим козырем
    assert classic_game.make_move(
        defender_id, 
        {'type': 'defend', 'move': [[attacker_id, '10S'], [defender_id, '9H']]}
    ) is False  # Должно вернуть False, так как защита невалидна

def test_transfer_in_transfer_game(transfer_game):
    """Тест перевода в переводном дураке"""
    transfer_game.trump = 'D'  # Для предсказуемости
    attacker_id = transfer_game.current_attacker
    defender_id = transfer_game.current_defender
    
    # Дадим игрокам карты одного номинала
    transfer_game.players[0]['hand'] = ['10H']  # Атакующий
    transfer_game.players[1]['hand'] = ['10S']  # Защитник
    
    # Атака
    transfer_game.make_move(attacker_id, {'type': 'attack', 'move': '10H'})
    
    # Перевод
    assert transfer_game.make_move(
        defender_id, 
        {'type': 'attack', 'move': '10S'}
    ) is True
    
    # Проверяем, что роли поменялись
    assert transfer_game.current_attacker == defender_id
    assert transfer_game.current_defender == attacker_id

def test_bat_action(classic_game):
    """Тест действия 'бито'"""
    classic_game.trump = 'C'
    attacker_id = classic_game.current_attacker
    defender_id = classic_game.current_defender
    
    classic_game.players[0]['hand'] = ['10H']  # Атакующий
    classic_game.players[1]['hand'] = ['12H']  # Защитник
    
    # Атака и защита
    classic_game.make_move(attacker_id, {'type': 'attack', 'move': '10H'})
    classic_game.make_move(
        defender_id, 
        {'type': 'defend', 'move': [[attacker_id, '10H'], [defender_id, '12H']]}
    )
    
    # Бито
    assert classic_game.make_move(attacker_id, {'type': 'state', 'state': 'bat'}) is True
    
    state = classic_game.get_game_state()
    assert len(state['table']) == 0  # Стол должен очиститься
    assert len(state['bat']) == 2   # В бито должно быть 2 карты

def test_game_end_condition(classic_game):
    """Тест условия окончания игры"""
    # Эмулируем конец игры
    classic_game.deck = []  # Опустошаем колоду
    classic_game.players[0]['hand'] = []  # У атакующего нет карт
    
    # Проверяем, что игра завершена
    classic_game._check_game_over()
    assert classic_game.game_over is True
    assert classic_game.winner == classic_game.players[0]['id']
    assert classic_game.players[0]['state'] == 'winner'
    assert classic_game.players[1]['state'] == 'durak'

def test_draw_cards(classic_game):
    """Тест добора карт"""
    # Оставим в колоде 4 карты
    classic_game.deck = ['9S', '10S', '11S', '12S']
    classic_game.players[0]['hand'] = ['10H']  # 1 карта
    classic_game.players[1]['hand'] = ['12H']  # 1 карта
    
    classic_game._draw_cards()
    
    # Проверяем, что карты добрались (но не более 6)
    assert len(classic_game.players[0]['hand']) == 5  # 1 было + 4 из колоды
    assert len(classic_game.players[1]['hand']) == 5
    assert len(classic_game.deck) == 0