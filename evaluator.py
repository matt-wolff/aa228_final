import torch
from model import PokerModel, RandomPokerModel
from poker import Game
import argparse
from tqdm import tqdm
import numpy as np
import random

NUM_GAMES = 1000
pbar = tqdm(total=NUM_GAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(42)
random.seed(42)

def main(model1_filename, model2_filename):
    if model1_filename == "random":
        model1 = RandomPokerModel()
    else:
        model1 = PokerModel(hidden_dim=1024, n_layers=5, dropout_p=0.5).to(DEVICE)
        model1.load_state_dict(torch.load(model1_filename, weights_only=True))
        model1.eval()

    if model2_filename == "random":
        model2 = RandomPokerModel()
    else:
        model2 = PokerModel(hidden_dim=1024, n_layers=5, dropout_p=0.5).to(DEVICE)
        model2.load_state_dict(torch.load(model2_filename, weights_only=True))
        model2.eval()

    num_finished = 0
    // players = [model1, model2]
    player_rewards_per_game = [[], []]

    game = Game()
    determine_position = random.randint(0,1)
    if determine_position:
        players = [model1, model2]
    else : 
        players = [model2, model1]
    
    next_state = game.get_state()
    reward_game = [0,0]
    with tqdm() as pbar:
        while num_finished != NUM_GAMES:
            current_player = game.current_to_act
            current_agent = players[current_player]
            current_state = next_state

            action, _ = current_agent.next_action(current_state, game, eval=True)
            reward, game_finished = game.perform_action(action)

            reward_game[current_player] += reward

            if game_finished:
                if determine_position:
                    player_rewards_per_game[current_player].append(reward_game[current_player])
                else:
                    player_rewards_per_game[other_player].append(reward_game[current_player])
                if reward != -1000:
                    other_player = 0 if current_player == 1 else 1
                    if reward < 0: #current_player lost, other_player won
                        other_r = game.pot - game.players[other_player].blind_bet
                    elif game.players[other_player].total_chips == 100: # players start with 100 chips, so if pot was split, player will still have 100 chips at end of game
                        other_r = (game.pot / 2) - game.players[other_player].blind_bet
                    else: # other player lost
                        other_r = -game.players[other_player].blind_bet
                    if determine_position:
                        player_rewards_per_game[other_player].append(reward_game[other_player] + other_r)
                    else:
                        player_rewards_per_game[current_player].append(reward_game[other_player] + other_r)

                game = Game()
                determine_position = random.randint(0,1)
                if determine_position:
                    players = [model1, model2]
                else : 
                    players = [model2, model1]
                reward_game = [0, 0]
                next_state = game.get_state()

                num_finished += 1
                pbar.update(1)
            else:
                next_state = game.get_state()
    
    print(f"Total rewards of model 1: {sum(player_rewards_per_game[0])}")
    print(f"Average rewards of model 1: {np.mean(player_rewards_per_game[0])}")
    print(f"Total rewards of model 2: {sum(player_rewards_per_game[1])}")
    print(f"Average rewards of model 2: {np.mean(player_rewards_per_game[1])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model1', '--m1', type=str, default="random")
    parser.add_argument('-model2', '--m2', type=str, default="models/self_play_20241128_133219_vanilla_False.pth")
    args = parser.parse_args()
    main(model1_filename=args.m1, model2_filename=args.m2)
