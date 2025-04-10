from typing import List, Dict, Tuple, Optional
import random

class DurakGame:
    # Статические переменные
    RANKS = ['9', '10', '11', '12', '13', '14']
    SUITS = ['S', 'C', 'D', 'H']  # Пики, Трефы, Бубны, Червы
    RANK_VALUES = {'9': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5}
    
    def __init__(self, game_type: int = 0):
        self.game_type = game_type
        self.deck = self._create_deck()
        self.trump = None
        self.players = []
        self.table = []
        self.bat = []
        self.current_attacker = None
        self.current_defender = None
        self.game_over = False
        self.winner = None
        
    def _create_deck(self) -> List[str]:
        return [rank + suit for suit in self.SUITS for rank in self.RANKS]
    
    def start_game(self, player_ids: List[str]):
        if len(player_ids) != 2:
            raise ValueError("Только 2 игрока поддерживается")
            
        random.shuffle(self.deck)
        self.trump = self.deck[-1][-1]  # Масть последней карты
        self.players = [
            {'id': player_ids[0], 'hand': [], 'state': 'wait'},
            {'id': player_ids[1], 'hand': [], 'state': 'wait'}
        ]
        
        # Раздаем карты
        for _ in range(6):
            for player in self.players:
                if self.deck:
                    player['hand'].append(self.deck.pop())
        
        self._determine_first_attacker()
        
    def _determine_first_attacker(self):
        min_trump_rank = float('inf')
        attacker_idx = 0
        
        for i, player in enumerate(self.players):
            for card in player['hand']:
                if card[-1] == self.trump:
                    rank_value = self.RANK_VALUES[card[:-1]]
                    if rank_value < min_trump_rank:
                        min_trump_rank = rank_value
                        attacker_idx = i
        
        self.current_attacker = self.players[attacker_idx]['id']
        self.current_defender = self.players[1 - attacker_idx]['id']
        
        self.players[attacker_idx]['state'] = 'attack'
        self.players[1 - attacker_idx]['state'] = 'defend'
    
    def make_move(self, player_id: str, move: Dict) -> bool:
        if self.game_over:
            return False
            
        player = next(p for p in self.players if p['id'] == player_id)
        
        if move['type'] == 'attack':
            return self._handle_attack(player, move)
        elif move['type'] == 'defend':
            return self._handle_defend(player, move)
        elif move['type'] == 'state':
            return self._handle_state(player, move)
        elif move['type'] == 'wait':
            return True
            
        return False
    
    def _handle_attack(self, player: Dict, move: Dict) -> bool:
        """Обрабатывает атаку или перевод"""
        if player['state'] not in ['attack', 'defend']:
            return False
            
        card = move['move']
        if card not in player['hand']:
            return False
            
        # Если игрок атакует
        if player['state'] == 'attack':
            # Первая атака - любая карта
            if not self.table:
                self.table.append({'attack_card': {'card': card, 'user_id': player['id']}})
                player['hand'].remove(card)
                return True
            # Последующие атаки - только карты того же достоинства
            else:
                table_ranks = {c['attack_card']['card'][:-1] for c in self.table}
                if card[:-1] in table_ranks:
                    self.table.append({'attack_card': {'card': card, 'user_id': player['id']}})
                    player['hand'].remove(card)
                    return True
                return False
                
        # Если игрок переводит (только для переводного дурака)
        elif player['state'] == 'defend' and self.game_type == 1:
            if len(self.table) == 1:  # Перевод возможен только при одной карте на столе
                attack_card = self.table[0]['attack_card']['card']
                if card[:-1] == attack_card[:-1]:  # Такое же достоинство
                    # Удаляем старую атаку и добавляем новую
                    self.table[0] = {'attack_card': {'card': card, 'user_id': player['id']}}
                    player['hand'].remove(card)
                    # Меняем роли
                    self._switch_roles()
                    return True
        return False
    
    def _handle_defend(self, player: Dict, move: Dict) -> bool:
        """Обрабатывает защиту"""
        if player['state'] != 'defend':
            return False
            
        # Проверяем все пары атака-защита
        for attack_defend in move['move']:
            attack_card, defend_card = attack_defend
            
            # Находим соответствующую атаку на столе
            table_attack = next(
                (a for a in self.table 
                 if a['attack_card']['card'] == attack_card and 'defend_card' not in a),
                None
            )
            
            if not table_attack:
                return False
                
            if not self._is_valid_defend(attack_card, defend_card, player):
                return False
                
            # Добавляем защиту
            table_attack['defend_card'] = {'card': defend_card, 'user_id': player['id']}
            player['hand'].remove(defend_card)
            
        return True
    
    def _is_valid_defend(self, attack_card: str, defend_card: str, player: Dict) -> bool:
        """Проверяет валидность защиты"""
        if defend_card not in player['hand']:
            return False
            
        attack_rank, attack_suit = attack_card[:-1], attack_card[-1]
        defend_rank, defend_suit = defend_card[:-1], defend_card[-1]
        
        # Козырь бьет некозырную карту
        if defend_suit == self.trump and attack_suit != self.trump:
            return True
            
        # Карта той же масти и старше
        if (defend_suit == attack_suit and 
            self.RANK_VALUES[defend_rank] > self.RANK_VALUES[attack_rank]):
            return True
            
        return False
    
    def _handle_state(self, player: Dict, move: Dict) -> bool:
        """Обрабатывает смену состояния (бито, пас, взять)"""
        state = move['state']
        
        if state == 'bat' and player['state'] == 'defend':
            # Все карты со стола в бито
            self._move_table_to_bat()
            self._end_round()
            return True
            
        elif state == 'take' and player['state'] == 'defend':
            # Защитник берет карты
            self._move_table_to_player(player)
            self._end_round()
            return True
            
        elif state == 'pass' and player['state'] == 'attack':
            # Атакующий завершает раунд
            self._end_round()
            return True
            
        return False
    
    def _switch_roles(self):
        """Меняет атакующего и защитника местами"""
        self.current_attacker, self.current_defender = self.current_defender, self.current_attacker
        for player in self.players:
            if player['id'] == self.current_attacker:
                player['state'] = 'attack'
            else:
                player['state'] = 'defend'
    
    def _move_table_to_bat(self):
        """Перемещает все карты со стола в бито"""
        for card_info in self.table:
            self.bat.append(card_info['attack_card']['card'])
            if 'defend_card' in card_info:
                self.bat.append(card_info['defend_card']['card'])
        self.table = []
    
    def _move_table_to_player(self, player: Dict):
        """Перемещает карты со стола игроку"""
        for card_info in self.table:
            player['hand'].append(card_info['attack_card']['card'])
            if 'defend_card' in card_info:
                player['hand'].append(card_info['defend_card']['card'])
        self.table = []
    
    def _end_round(self):
        """Завершает раунд и начинает новый"""
        # Добираем карты
        self._draw_cards()
        
        # Проверяем условия окончания игры
        self._check_game_over()
        
        if not self.game_over:
            # Меняем роли для нового раунда
            self._switch_roles()
    
    def _draw_cards(self):
        """Добирает карты игрокам до 6 из колоды"""
        for player in self.players:
            while len(player['hand']) < 6 and self.deck:
                player['hand'].append(self.deck.pop())
    
    def _check_game_over(self):
        """Проверяет условия окончания игры"""
        # Если у одного игрока закончились карты и колода пуста
        for player in self.players:
            if not player['hand'] and not self.deck:
                self.game_over = True
                self.winner = player['id']
                player['state'] = 'winner'
                self.players[1 - self.players.index(player)]['state'] = 'durak'
                return
                
        # Если в колоде нет карт и защитник взял последние карты
        if not self.deck and any(p['state'] == 'take' for p in self.players):
            self.game_over = True
            # Победитель тот, кто не взял карты
            winner = next(p for p in self.players if p['state'] != 'take')
            self.winner = winner['id']
            winner['state'] = 'winner'
            self.players[1 - self.players.index(winner)]['state'] = 'durak'
    
    def get_game_state(self) -> Dict:
        """Возвращает текущее состояние игры в формате JSON"""
        return {
            'game_id': 'current_game',  # В реальной игре нужно генерировать UUID
            'trump': self.trump,
            'timestamp': 0,  # Нужно вести счетчик ходов
            'winner': self.winner,
            'game_rules': {
                'game_type': self.game_type,
                'player_amount': 2,
                'card_amount': 24
            },
            'deck': self.deck.copy(),
            'bat': self.bat.copy(),
            'table': self.table.copy(),
            'players': [
                {
                    'id': p['id'],
                    'state': p['state'],
                    'hand': p['hand'].copy()
                } for p in self.players
            ]
        }