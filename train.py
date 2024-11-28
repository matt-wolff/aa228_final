import torch
from model import PokerModel
from poker import Game
import copy
import random
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

NUM_ACTIONS = 10000  # Number of actions to take
BATCH_SIZE = 10
GAMMA = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot_losses(losses, timestamp, is_vanilla):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if is_vanilla:
        title = 'Vanilla Self Play Loss'
    else:
        title = "Dealer's Choice Self Play Loss"
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"plots/self_play_{timestamp}_vanilla_{is_vanilla}.png")

def main(is_vanilla):
    agent = PokerModel(hidden_dim=1024, n_layers=5, dropout_p=0.5).to(DEVICE)
    target_agent = agent

    game = Game()
    replay_buffer = []
    losses = []

    transtion_construction = [
        [],
        []
    ]

    next_state = game.get_state()
    for i in range(NUM_ACTIONS):
        current_player = game.current_to_act
        current_state = next_state
        if transtion_construction[current_player]:
            transtion_construction[current_player].append(current_state)
            if len(replay_buffer) >= 100:
                    replay_buffer = replay_buffer[1:] + [transtion_construction[current_player]]
            else:
                replay_buffer.append(transtion_construction[current_player])
            transtion_construction[current_player] = []

        action, _ = agent.next_action(current_state)
        reward, game_finished = game.perform_action(action)

        if game_finished:
            cur_player_transition = [current_state, action, reward, None]
            if len(replay_buffer) >= 100:
                replay_buffer = replay_buffer[1:] + [cur_player_transition]
            else:
                replay_buffer.append(cur_player_transition)

            if reward != -1000:  # if opponent didn't end game by taking illegal action
                other_player = 0 if current_player == 1 else 1
                if transtion_construction[other_player]:
                    other_s, other_a, _ = transtion_construction[other_player]
                    other_r = -1 * (game.pot - reward)
                    replay_buffer.append([other_s, other_a, other_r, None])
            
            transtion_construction = [[],[]]
            game = Game()
            next_state = game.get_state()
        else:
            next_state = game.get_state()
            next_player = game.current_to_act
            if next_player != current_player:
                transtion_construction[current_player] = [current_state, action, reward]
            else:
                if len(replay_buffer) >= 100:
                    replay_buffer = replay_buffer[1:] + [[current_state, action, reward, next_state]]
                else:
                    replay_buffer.append([current_state, action, reward, next_state])
        
        if len(replay_buffer) >= BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)

            loss = 0
            for transition in minibatch:
                s_t, a_t, r_t, s_t_plus_one = transition
                if s_t_plus_one == None:
                    y = r_t
                else:
                    with torch.no_grad():
                        _, target_val = target_agent.max_action(s_t_plus_one)
                        y = r_t + GAMMA * target_val
                loss += (y - agent.model(s_t)[a_t]) ** 2

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            print(f"Iteration: {i}, Loss: {loss.float()}")
            losses.append(loss.float())

        if (i+1 % 1000 == 0):
            target_agent = copy.deepcopy(agent)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_losses(losses, timestamp, is_vanilla)
    torch.save(agent.state_dict(), f"models/self_play_{timestamp}_vanilla_{is_vanilla}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vanilla', action="store_true")
    args = parser.parse_args()
    main(is_vanilla=args.vanilla)
