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

#hand evaluation comparison function, define different types of hands, need to figure out how to use both table card and hole cards, check for all possible combinations. only need to do this at the end if there is showdown

# players listed in seating order, button is index of seating order and will increment
#our_player can be index of player that our model is tracking
                  
#players listed in seated order, button is index for dealer
#dealers choice rule is array for wild card
class Game:
    # create deck function, 11-14 face cards (10,11,12,13,14)
    # one-hot encoding, contantenated
    # create deck function, 11-14 face cards (10,11,12,13,14)
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
    
    def __init__(self, names=["Tomas", "Mattheus"], big_blind=2, small_blind=1, button=0, dealers_choice_rule=np.zeros(13)):
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
        self.dealers_choice_rule = dealers_choice_rule
        self.pot = 0
        self.current_to_act = self.update_current_to_act(self.button, self.num_players)
        self.bet_to_call = big_blind

        #need to initialize so it starts at first state with action (after blinds)

        #deal cards
        for player in self.players:
            player.hole_cards = [self.deck.pop() for i in range(2)]
        
        #pay blinds
        #small blinds
        self.players[self.current_to_act].total_chips -= self.small_blind
        self.update_current_to_act(self.current_to_act, self.num_players)
        self.pot += self.small_blind

        # big blind
        self.players[self.current_to_act].total_chips -=  self.big_blind
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

     # funtion which returns reward for current player of showdown
    def showdown(self, reward):
        converted_button = [Card.new(self.convert_card(card)) for card in self.players[0].hole_cards]
        converted_other = [Card.new(self.convert_card(card)) for card in self.players[1].hole_cards]
        separated_tabled_cards = [self.table_cards[i*17:(i+1)*17] for i in range(5)]
        converted_table = [Card.new(self.convert_card(card)) for card in separated_tabled_cards]

        evaluator = Evaluator()
        button_score = evaluator.evaluate(converted_table, converted_button)
        other_score = evaluator.evaluate(converted_table, converted_other)

        #remember lower score is better
        if ((button_score - other_score) > 0): #other wins
            if self.current_to_act == 1:
                return self.pot
            else:
                return reward
        elif ((button_score - other_score) < 0): #button wins
            if self.current_to_act == 0:
                return self.pot
            else:
                return reward
        else: #split pot
            return self.pot / 2
    
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
        self.bet_to_call = amount
    
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
        reward = -self.players[self.current_to_act].total_chips
        # if all-in is just a call (do self.call)
        if (self.bet_to_call >= -reward):
            self.bet(-reward)
            return self.runout_game(reward)
        else: #is a raise, need to see if opponent calls
            self.bet(-reward)
            self.current_to_act = self.update_current_to_act(self.current_to_act, self.num_players)
            return reward, False

    def raise_bet(self, amount):
        reward = -amount
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
        current_player = self.players[self.current_to_act] # remember this is only a copy
        if (action == 0): # check
            return self.check()
        elif (action == 1): # call
            return self.call()
        elif (action == 2): # fold
            return 0, True
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
