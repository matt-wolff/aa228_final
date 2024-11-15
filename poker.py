import pandas as pd
import numpy as np
import random



class player:
    def __init__(self, name, total_chips, total_chips_bought, policy):
        self.name = name
        self.total_chips = total_chips
        self.total_chips_bought = total_chips_bought
        self.hole_cards = []
        self.policy = policy

    def add_money(self, amount_adding):
        self.total_chips += amount_adding
        self.total_chips_bought += amount_adding

#hand evaluation comparison function, define different types of hands, need to figure out how to use both table card and hole cards, check for all possible combinations. only need to do this at the end if there is showdown

# players listed in seating order, button is index of seating order and will increment
#our_player can be index of player that our model is tracking
class game:
    def __init__(self, players, big_blind, small_blind, our_player, button):
        self.players = players
        self.our_player = our_player
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.seating_order = seating_order
        self.button = button

#players listed in seated order, button is index for dealer
#dealers choice rule is array for wild card
class hand:
    #create deck function, 11-14 face cards (10,11,12,13,14)
    # need to change to one-hot encoding, contantenated
    def create_deck(self):
        # Define ranks and suits
        values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades'] # catetorical for training
        # list of tuples (value, suit)
        deck = [(value, suit) for value in values for suit in suits]

        random.shuffle(deck)

        return deck

    def update_current_to_act(self, index_to_act, n_players): #needed before init
        index_to_act++
        if (index_to_act >= n_players):
            return 0
        return index_to_act

    def __init__(self, game, dealers_choice_rule):
        self.game = game
        self.num_players = len(game.players)
        self.deck = create_deck()
        self.table_cards = []
        self.currrent_round = "pre-flop"
        self.dealers_choice_rule = dealers_choice_rule
        self.pot = pot
        self.current_to_act = self.update_current_to_act(self.button, self.num_players)

        #need to initialize so it starts at our player's first state
        #pay blinds
        #small blind
        game.players[current_to_act].total_chips -= game.small.blind
        if (
        self.update_current_to_act(self.current_to_act, self.num_players)
        pot += game.small_blind

    def update_current_to_act(self):
        self.current_to_act++
        if(self.current_to_act >= len(players)):
            self.current_to_act = 0

    def showdown(self):

    #returns current state (hole cards, table cards, current bet to call, current pot, current money available) note: discuss with matt about who calling / betting history as a state, right now is heads up so who cares
    #this is function matt will call while training to return the
    # I need to give the state, matt gives me the action, I return the reward and next state
    #get_state which returns current state
    #get_reward which returns current rewards when given action and leaves game at next state
    def get_state(self):

        return (self.dealers_choice_rule)

    def get_reward(self, action):

