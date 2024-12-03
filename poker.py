import pandas as pd
import numpy as np
import random
import torch
from treys import Card
from treys import Evaluator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class Player:
    def __init__(self, name, total_chips=100):
        self.name = name
        self.total_chips = total_chips
        self.hole_cards = []
        self.blind_bet = 0
        self.current_round_bet = 0

# players listed in seating order, button is index of seating order and will increment
#our_player can be index of player that our model is tracking
                  
#players listed in seated order, button is index for dealer
#dealers choice rule is array for wild card
class Game:
    # create deck function, face cards 10,11,12,13,14
    # one-hot encoding, contantenated
    def create_deck(self):
        # Define ranks and suits
        values = np.identity(13)
        suits = np.identity(4) # catetorical for training
        deck = []
        for i in range(len(values)):
            for j in range(len(suits)):
                deck.append(np.concatenate((values[i], suits[j])))

        random.shuffle(deck)

        return deck

    def update_current_to_act(self, index_to_act, n_players): #needed before init
        index_to_act += 1
        if (index_to_act >= n_players):
            return 0
        return index_to_act
    
    def __init__(self, names=["Tomas", "Mattheus"], big_blind=2, small_blind=1, button=0, vanilla=True, bomb=False):
        self.players = []
        for name in names: # i did this the fun way, probably a better way if you want to make class more generalizable
            self.players.append(Player(name, 100))
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.button = button
        self.num_players = len(self.players)
        self.deck = self.create_deck()
        self.table_cards = np.zeros(17 * 5)
        self.current_round = 0 # 0 = pre-flop, 1 = flop, and so on...
        self.pot = 0
        self.current_to_act = self.update_current_to_act(self.button, self.num_players)
        self.bet_to_call = big_blind
        #set dealers choice rules using vanilla boolean (no wild card if vanilla == True)
        self.dealers_choice_rule = np.zeros(14)
        self.wild_card = np.zeros(13)
        if vanilla == False:
            #set random wild card
            wild_index = random.randint(0, 12)
            self.dealers_choice_rule[wild_index] = 1
            self.wild_card[wild_index] = 1
        
        #need to initialize game so it starts at the first state with action (after blinds)

        #deal cards
        for player in self.players:
            player.hole_cards = [self.deck.pop() for i in range(2)]
        
        #pay blinds or bomb

        if bomb:
            self.bomb_amount = 10
            self.dealers_choice_rule[13] = 1
            #players pay bomb pot
            self.players[0].total_chips -= self.bomb_amount
            self.players[0].blind_bet += self.bomb_amount
            self.pot += self.bomb_amount

            self.players[1].total_chips -=  self.bomb_amount
            self.players[1].blind_bet += self.bomb_amount
            self.pot += self.bomb_amount
            
            #next round (straight to flop)
            _, _ = self.next_round(0)
        else:    
            #small blinds
            self.players[self.current_to_act].total_chips -= self.small_blind
            self.players[self.current_to_act].blind_bet += self.small_blind
            self.update_current_to_act(self.current_to_act, self.num_players)
            self.pot += self.small_blind

            # big blind
            self.players[self.current_to_act].total_chips -=  self.big_blind
            self.players[self.current_to_act].blind_bet += self.big_blind
            self.update_current_to_act(self.current_to_act, self.num_players)
            self.pot += self.big_blind

        #now we are good to goooo baby

    #YO! YOU FROM BROOKLYN?
    #get_state which returns current state
    # returns tuple of dealers choice rule (array), hole cards (array of arrays), table cards (array of arrays), pot, bet to call, and position
    def get_state(self):
        current_player = self.players[self.current_to_act]

        # TODO: Figure out where it is becoming a tensor
        if isinstance(self.pot, torch.Tensor):
            self.pot = self.pot.cpu()
        if isinstance(self.bet_to_call, torch.Tensor):
            self.bet_to_call = self.bet_to_call.cpu()
        if isinstance(current_player.total_chips, torch.Tensor):
            current_player.total_chips = current_player.total_chips.cpu()

        state = np.concatenate(
            [self.dealers_choice_rule] + current_player.hole_cards + [self.table_cards] + [np.array([self.pot, self.bet_to_call, current_player.total_chips, self.current_to_act])]
        )
        return torch.from_numpy(state).float().to(DEVICE)
    
    def convert_card(self, card):
        indexes = [i for i, x in enumerate(card) if x == 1]
        values_dict = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        value = indexes[0] + 2
        if (value >= 10):
            value = values_dict[value]
        
        suit_dict = {13: "h", 14: "d", 15: "c", 16: "s"}
        suit = suit_dict[indexes[1]]

        return str(value) + suit

    def evaluate_hand(self, player_index):
        separated_tabled_cards = [self.table_cards[i*17:(i+1)*17] for i in range(5)]
        evaluator = Evaluator()
        
        #check if there is wild_card rule active
        if 1 in self.wild_card:
            wild_index = self.wild_card.index(1)
            
            #for looping through possible cards
            all_possible_cards = self.create_deck()
            
            hole_wild_count = 0
            converted_hole = []
            for card in self.players[player_index].hole_cards:
                if card[wild_index] == 0:
                    converted_hole.append(self.convert_card(card))
                else:
                    hole_wild_count += 1
            
            converted_hole_combinations = [] #use to store all combinations of cards so we can try together with all combinations of table cards
            # I could've made a recursive function to build to combinations lists...but this seemed easier although messier
            if hole_wild_count == 0:
                converted_hole_combinations = [converted_hole]
            else:
                for card1 in all_possible_cards:
                    if hole_wild_count == 2:
                        for card2 in all_possible_cards:
                            converted_hole_combinations.append([self.convert_card(card1), self.convert_card(card2)]) 
                    else: 
                        converted_hole_combinations.append([converted_hole[0], self.convert_card(card1)])
            
            # do the same for table cards
            table_wild_count = 0
            converted_table = []
            for card in separated_tabled_cards:
                if card[wild_index] == 0:
                    converted_table.append(self.convert_card(card))
                else:
                    table_wild_count += 1
            
            converted_table_combinations = []
            if table_wild_count == 0:
                converted_table_combinations = [converted_table]
            else:
                for card1 in all_possible_cards:
                    if table_wild_count > 1:
                        for card2 in all_possible_cards:
                            if table_wild_count > 2:
                                for card3 in all_possible_cards:
                                    if table_wild_count == 4:
                                        for card4 in all_possible_cards:
                                            converted_table_combinations.append(converted_table + [self.convert_card(card1), self.convert_card(card2), self.convert_card(card3), self.convert_card(card4)])
                                    else:
                                        converted_table_combinations.append(converted_table + [self.convert_card(card1), self.convert_card(card2), self.convert_card(card3)])
                            else:
                                converted_table_combinations.append(converted_table + [self.convert_card(card1), self.convert_card(card2)])
                    else: 
                        converted_table_combinations.append(converted_table + [self.convert_card(card1)])
            
            best_score = 7462  # 7462 is the worst score from Treys evaluator
            for hole in converted_hole_combinations:
                for table in converted_table_combinations:
                    hand_score = evaluator.evaluate(table, hole)
                    if hand_score < best_score:
                        best_score = hand_score
            
            return best_score
        else:
            #no wild cards, do normal conversions
            converted_hole = [Card.new(self.convert_card(card)) for card in self.players[player_index].hole_cards]
            converted_table = [Card.new(self.convert_card(card)) for card in separated_tabled_cards]
            return evaluator.evaluate(converted_table, converted_hole)
        pass    

     # funtion which returns reward for current player of showdown
    def showdown(self, reward):

        button_score = self.evaluate_hand(0)
        other_score = self.evaluate_hand(1)
        
        #remember lower score is better
        #blinds get factored in at the end of the game
        if ((button_score - other_score) > 0): #other wins
            if self.current_to_act == 1:
                self.players[1].total_chips += self.pot
                return self.pot + reward - self.players[1].blind_bet
            else:
                return reward  - self.players[1].blind_bet
        elif ((button_score - other_score) < 0): #button wins
            if self.current_to_act == 0:
                self.players[0].total_chips += self.pot
                return self.pot + reward - self.players[0].blind_bet
            else:
                return reward  - self.players[0].blind_bet
        else: #split pot
            self.players[0].total_chips += int(self.pot / 2)
            self.players[1].total_chips += int(self.pot / 2)
            return (self.pot / 2) + reward - self.players[self.current_to_act].blind_bet
    
    def next_round(self, reward):
        if self.current_round == 0: #pre-flop
            for i in range(3):
                self.table_cards[i*17:(i+1)*17] = self.deck.pop()
        elif self.current_round == 1:  # flop
            self.table_cards[3*17:(3+1)*17] = self.deck.pop()
        elif self.current_round == 2:  # turn
            self.table_cards[4*17:(4+1)*17] = self.deck.pop()
        elif self.current_round == 3: #river
            return self.showdown(reward), True

        self.current_to_act = self.update_current_to_act(self.button, self.num_players)
        self.current_round += 1
        self.bet_to_call = 0
        for player in self.players:
            player.current_round_bet = 0
        return reward, False

    #give rest of table cards and then run showdown
    def runout_game(self, reward):
        copy_cur = self.current_to_act
        while (self.current_round < 3):
            self.next_round(reward)

        self.current_to_act = copy_cur
        return self.next_round(reward)

    def bet(self, amount):
        if isinstance(amount, torch.Tensor):
            amount = amount.cpu()
        self.pot += amount 
        self.players[self.current_to_act].total_chips -= amount
        self.players[self.current_to_act].current_round_bet += amount
        self.bet_to_call = amount + self.players[self.current_to_act].current_round_bet
    
    def check(self):
        reward = 0
        # if check is illegal, reward is super negative and end game/explode players head
        if (self.bet_to_call > 0):
            return -1000, True
        elif (self.current_to_act == 0):
            return self.next_round(reward)
        else:
            self.current_to_act = self.update_current_to_act(self.current_to_act, self.num_players)
            return reward, False
        
    def call(self):
         reward = -self.bet_to_call
         #if call leaves player broke
         if (self.bet_to_call >= self.players[self.current_to_act].total_chips):  
             self.bet(self.players[self.current_to_act].total_chips)
             return self.runout_game(reward) # TOD0: need to adjust the pot, didn't match full bet, (wait, doesn't matter here? heads up with same starting money, shouldn't be possible
         elif (self.current_to_act == 0):
             self.bet(-reward)
             return self.next_round(reward)
         else:
             self.bet(-reward)
             self.current_to_act = self.update_current_to_act(self.current_to_act, self.num_players)
             return reward, False

    def all_in(self):
        reward = -self.players[self.current_to_act].total_chips + self.players[self.current_to_act].current_round_bet
        # if all-in is just a call (do self.call)
        if (self.bet_to_call >= -reward):
            self.bet(-reward)
            return self.runout_game(reward)
        else: #is a raise, need to see if opponent calls
            self.bet(-reward)
            self.current_to_act = self.update_current_to_act(self.current_to_act, self.num_players)
            return reward, False

    def raise_bet(self, amount):
        reward = -amount + self.players[self.current_to_act].current_round_bet
        #check for illegal actions (not min raise)
        if (amount <= (2 * self.bet_to_call)):
            return -1000, True
        #if raise amount is basically just an all-in
        elif (amount >= self.players[self.current_to_act].total_chips):
            return self.all_in()
        else:
            self.bet(amount)
            self.current_to_act = self.update_current_to_act(self.current_to_act, self.num_players)
            return reward, False
        
    # final reward - final if no money left: finish the game, showdown, and then game_finsished
    # performs only one action, updates current_to_act
    def perform_action(self, action):
        #return reward and boolean of 'game_finished'
        if (action == 0): # check
            return self.check()
        elif (action == 1): # call
            return self.call()
        elif (action == 2): # fold
            return -self.players[self.current_to_act].blind_bet, True
        elif (action == 3 or action == 4 or action == 5): # raise bet_to_call (2x, 3x, 4x)
            raise_scalar = action - 1
            amount = self.bet_to_call * raise_scalar
            return self.raise_bet(amount)
        elif (action == 6 or action == 7 or action == 8 or action == 9): # raise by the pot (1/4, 1/2, 1 ,2) Q: 4 also?
            raise_scalar = 2 ** (action - 6)
            pot_scalar = (1/4) * raise_scalar
            amount = int(pot_scalar * self.pot)
            return self.raise_bet(amount)
        else: #all-in
            return self.all_in()
    
    # available actions for the current player
    def available_actions(self):
        actions = [1, 2, 10]  # calling, folding, and going all-in are always legal

        if (self.bet_to_call == 0):  # if checking is legal
            actions.append(0)

        def raising_call_legal(action):
            raise_scalar = action - 1
            amount = self.bet_to_call * raise_scalar
            return amount > 2 * self.bet_to_call
        if (raising_call_legal(3)):  # raise call 2x
            actions.append(3)
        if (raising_call_legal(4)):  # raise call 3x
            actions.append(4)
        if (raising_call_legal(5)):  # raise call 4x
            actions.append(5)
        
        def raising_pot_legal(action):
            raise_scalar = 2 ** (action - 6)
            pot_scalar = (1/4) * raise_scalar
            amount = int(pot_scalar * self.pot)
            return amount > 2 * self.bet_to_call
        if (raising_pot_legal(6)):  # raise pot 0.25x
            actions.append(6)
        if (raising_pot_legal(7)):  # raise 0.5x
            actions.append(7)
        if (raising_pot_legal(8)):  # raise 1x
            actions.append(8)
        if (raising_pot_legal(9)):  # raise 2x
            actions.append(9)
        
        return actions
