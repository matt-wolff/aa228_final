import torch
from model import PokerModel
from poker import Game
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

NUM_GAMES = 1000
pbar = tqdm(total=NUM_GAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(model1_filename, model2_filename):
    model1 = PokerModel(hidden_dim=1024, n_layers=5, dropout_p=0.5).to(DEVICE)
    model1.load_state_dict(torch.load(model1_filename, weights_only=True))
    model1.eval()

    model2 = PokerModel(hidden_dim=1024, n_layers=5, dropout_p=0.5).to(DEVICE)
    model2.load_state_dict(torch.load(model2_filename, weights_only=True))
    model2.eval()

    num_finished = 0
    players = [model1, model2]
    player_rewards_per_game = [[], []]

    game = Game()
    next_state = game.get_state()
    with tqdm() as pbar:
        while num_finished != NUM_GAMES:
            current_player = game.current_to_act
            current_agent = players[current_player]
            current_state = next_state

            action, _ = current_agent.next_action(current_state)
            reward, game_finished = game.perform_action(action)

            if game_finished:
                player_rewards_per_game[current_player].append(reward)
                other_player = 0 if current_player == 1 else 1
                player_rewards_per_game[other_player].append(-1* (game.pot - reward))

                game = Game()
                next_state = game.get_state()

                num_finished += 1
                pbar.update(1)
            else:
                next_state = game.get_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model1', '--m1', type=str)
    parser.add_argument('-model2', '--m2', type=str)
    args = parser.parse_args()
    main(model1_filename=args.m1, model2_filename=args.m2)
