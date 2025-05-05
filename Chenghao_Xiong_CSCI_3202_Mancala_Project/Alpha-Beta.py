import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy
import multiprocessing
import asyncio
from datetime import datetime

random.seed()

class Mancala:
    def __init__(self, pits_per_player=6, stones_per_pit=4):
        """
        The constructor for the Mancala class defines several instance variables:

        pits_per_player: This variable stores the number of pits each player has.
        stones_per_pit: It represents the number of stones each pit contains at the start of any game.
        board: This data structure is responsible for managing the Mancala board.
        current_player: This variable takes the value 1 or 2, as it's a two-player game, indicating which player's turn it is.
        moves: This is a list used to store the moves made by each player. It's structured in the format (current_player, chosen_pit).
        p1_pits_index: A list containing two elements representing the start and end indices of player 1's pits in the board data structure.
        p2_pits_index: Similar to p1_pits_index, it contains the start and end indices for player 2's pits on the board.
        p1_mancala_index and p2_mancala_index: These variables hold the indices of the Mancala pits on the board for players 1 and 2, respectively.
        """
        self.pits_per_player = pits_per_player
        self.board = [stones_per_pit] * ((pits_per_player + 1) * 2)  # Initialize each pit with stones_per_pit number of stones
        self.players = 2
        self.current_player = 1
        self.moves = []
        self.p1_pits_index = [0, self.pits_per_player - 1]
        self.p1_mancala_index = self.pits_per_player
        self.p2_pits_index = [self.pits_per_player + 1, len(self.board) - 1 - 1]
        self.p2_mancala_index = len(self.board) - 1

        # Zeroing the Mancala for both players
        self.board[self.p1_mancala_index] = 0
        self.board[self.p2_mancala_index] = 0

        # Record the number of turns
        self.turns = {"p1": 0, "p2": 0}

    def display_board(self):
        """
        Displays the board in a user-friendly format
        """
        player_1_pits = self.board[self.p1_pits_index[0] : self.p1_pits_index[1] + 1]
        player_1_mancala = self.board[self.p1_mancala_index]
        player_2_pits = self.board[self.p2_pits_index[0] : self.p2_pits_index[1] + 1]
        player_2_mancala = self.board[self.p2_mancala_index]

        print("P1               P2")
        print("     ____{}____     ".format(player_2_mancala))
        for i in range(self.pits_per_player):
            if i == self.pits_per_player - 1:
                print(
                    "{} -> |_{}_|_{}_| <- {}".format(i + 1, player_1_pits[i], player_2_pits[-(i + 1)], self.pits_per_player - i)
                )
            else:
                print(
                    "{} -> | {} | {} | <- {}".format(i + 1, player_1_pits[i], player_2_pits[-(i + 1)], self.pits_per_player - i)
                )

        print("         {}         ".format(player_1_mancala))
        turn = "P1" if self.current_player == 1 else "P2"
        print("Turn: " + turn)

    def valid_move(self, pit):
        """
        Function to check if the pit chosen by the current_player is a valid move.
        """

        # write your code here
        if self.current_player == 1:
            pit_index = pit - 1
            valid_range = range(self.p1_pits_index[0], self.p1_pits_index[1] + 1)
        else:
            pit_index = self.p2_pits_index[0] + (pit - 1)
            valid_range = range(self.p2_pits_index[0], self.p2_pits_index[1] + 1)

        if pit_index not in valid_range or self.board[pit_index] == 0:
            print("Invalid move\n")
            return False

        self.actual_pit_index = pit_index
        return True

    def random_move_generator(self):
        """
        Function to generate random valid moves with non-empty pits for the random player
        """

        # write your code here
        if self.current_player == 1:
            pits_range = range(self.p1_pits_index[0], self.p1_pits_index[1] + 1)
        else:
            pits_range = range(self.p2_pits_index[0], self.p2_pits_index[1] + 1)
        non_empty = [i for i in pits_range if self.board[i] > 0]
        if not non_empty:
            return None

        chosen_board_index = random.choice(non_empty)

        if self.current_player == 1:
            return chosen_board_index - self.p1_pits_index[0] + 1
        else:
            return chosen_board_index - self.p2_pits_index[0] + 1

    def play(self, pit):
        """
        This function simulates a single move made by a specific player using their selected pit. It primarily performs three tasks:
        1. It checks if the chosen pit is a valid move for the current player. If not, it prints "INVALID MOVE" and takes no action.
        2. It verifies if the game board has already reached a winning state. If so, it prints "GAME OVER" and takes no further action.
        3. After passing the above two checks, it proceeds to distribute the stones according to the specified Mancala rules.

        Finally, the function then switches the current player, allowing the other player to take their turn.
        """

        # write your code here
        if not self.valid_move(pit):
            return
        self.turns[f"p{self.current_player}"] += 1
        # 核心：这里直接打印"Player X chose pit Y"
        # print(f"Player {self.current_player} chose pit: {pit}")

        pit_index = self.actual_pit_index
        stones = self.board[pit_index]
        self.board[pit_index] = 0
        current_index = pit_index

        while stones > 0:
            current_index = (current_index + 1) % len(self.board)
            # Skip opponent's mancala
            if self.current_player == 1 and current_index == self.p2_mancala_index:
                continue
            if self.current_player == 2 and current_index == self.p1_mancala_index:
                continue
            self.board[current_index] += 1
            stones -= 1


        if self.board[current_index] == 1:
            if self.current_player == 1 and current_index in range(self.p1_pits_index[0], self.p1_pits_index[1] + 1):
                opposite_index = self.p2_pits_index[1] - (current_index - self.p1_pits_index[0])
                captured = self.board[opposite_index]
                if captured > 0:
                    self.board[self.p1_mancala_index] += captured + 1
                    self.board[opposite_index] = 0
                    self.board[current_index] = 0

            elif self.current_player == 2 and current_index in range(self.p2_pits_index[0], self.p2_pits_index[1] + 1):
                opposite_index = self.p1_pits_index[1] - (current_index - self.p2_pits_index[0])
                captured = self.board[opposite_index]
                if captured > 0:
                    self.board[self.p2_mancala_index] += captured + 1
                    self.board[opposite_index] = 0
                    self.board[current_index] = 0
        
        
        self.moves.append((self.current_player, pit))
        self.current_player = 2 if self.current_player == 1 else 1

    def winning_eval(self):
        """
        Function to verify if the game board has reached the winning state.
        Hint: If either of the players' pits are all empty, then it is considered a winning state.
        """
        # write your code here
        # Check if any side is empty
        p1_empty = all(self.board[i] == 0 for i in range(self.p1_pits_index[0], self.p1_pits_index[1] + 1))
        p2_empty = all(self.board[i] == 0 for i in range(self.p2_pits_index[0], self.p2_pits_index[1] + 1))
        if not p1_empty and not p2_empty:
            return None  # Game not over yet

        # Collect remaining stones for the non-empty player
        if p1_empty:
            for i in range(self.p2_pits_index[0], self.p2_pits_index[1] + 1):
                self.board[self.p2_mancala_index] += self.board[i]
                self.board[i] = 0
        elif p2_empty:
            for i in range(self.p1_pits_index[0], self.p1_pits_index[1] + 1):
                self.board[self.p1_mancala_index] += self.board[i]
                self.board[i] = 0

        # Calculate scores
        p1_score = self.board[self.p1_mancala_index]
        p2_score = self.board[self.p2_mancala_index]

        print(f"Game Over! P1 Score: {p1_score}, P2 Score: {p2_score}")
        if p1_score > p2_score:
            print("Player 1 wins!")
            return 1
        elif p2_score > p1_score:
            print("Player 2 wins!")
            return 2
        else:
            print("It's a tie!")
            return 0
        
        

class AlphaBetaPlayer:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    @staticmethod
    def utility(game: Mancala):
        return game.board[game.p2_mancala_index] - game.board[game.p1_mancala_index]
    
    @staticmethod
    def utility_v2(game: Mancala):
        # 权重可以调整，比如仓库权重为1，pits权重为0.3
        mancala_weight = 1.0
        pits_weight = 0.3

        p2_score = game.board[game.p2_mancala_index]
        p1_score = game.board[game.p1_mancala_index]
        p2_pits = sum(game.board[game.p2_pits_index[0] : game.p2_pits_index[1] + 1])
        p1_pits = sum(game.board[game.p1_pits_index[0] : game.p1_pits_index[1] + 1])

        return mancala_weight * (p2_score - p1_score) + pits_weight * (p2_pits - p1_pits)
    

    @staticmethod
    def is_game_over(game):
        p1_empty = all(game.board[i] == 0 for i in range(game.p1_pits_index[0], game.p1_pits_index[1] + 1))
        p2_empty = all(game.board[i] == 0 for i in range(game.p2_pits_index[0], game.p2_pits_index[1] + 1))
        return p1_empty or p2_empty

    def get_next_move(self, game):
        _, best_move = self.alphabeta(game, depth=0, alpha=float("-inf"), beta=float("inf"), maximizing=True)
        return best_move

    def alphabeta(self, game, depth, alpha, beta, maximizing):
        if depth == self.max_depth or self.is_game_over(game):
            return self.utility(game), None

        if maximizing:
            max_eval = float("-inf")
            best_move = None
            for move in self.valid_moves(game, 2):  # P2 是 Alpha-Beta 玩家
                next_game = copy.deepcopy(game)
                next_game.play(move)
                eval, _ = self.alphabeta(next_game, depth + 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            best_move = None
            for move in self.valid_moves(game, 1):  # P1 是随机玩家
                next_game = copy.deepcopy(game)
                next_game.play(move)
                eval, _ = self.alphabeta(next_game, depth + 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def valid_moves(self, game, player):
        if player == 1:
            pits_range = range(game.p1_pits_index[0], game.p1_pits_index[1] + 1)
        else:
            pits_range = range(game.p2_pits_index[0], game.p2_pits_index[1] + 1)
        return [i - pits_range.start + 1 for i in pits_range if game.board[i] > 0]


def do_one_alphabeta_game(verbose=False, depth=5):
    start_time = datetime.now()
    game = Mancala(pits_per_player=6, stones_per_pit=4)
    ab_player = AlphaBetaPlayer(max_depth=depth)

    while True:
        # 随机玩家 P1
        move = game.random_move_generator()
        game.play(move)
        if verbose:
            game.display_board()
        if game.winning_eval() is not None:
            break

        # Alpha-Beta 玩家 P2
        move = ab_player.get_next_move(game)
        if move is not None:
            game.play(move)
        if verbose:
            game.display_board()
        if game.winning_eval() is not None:
            break

    winner = game.winning_eval()
    end_time = datetime.now()
    return winner, game.turns["p2" if winner == 2 else "p1"], end_time - start_time


def run_alphabeta_games_parallel(n_games=100, ab_depth=10):
    avg_runtime = []
    with multiprocessing.Pool() as pool:
        results_async = [pool.apply_async(do_one_alphabeta_game, kwds={"verbose": False, "depth": ab_depth}) for _ in range(n_games)]
        results = [r.get() for r in tqdm(results_async)]
    random_wins = sum(1 for r in results if r[0] == 1)
    ab_wins = sum(1 for r in results if r[0] == 2)
    draws = sum(1 for r in results if r[0] == 0)
    avg_turns = sum(r[1] for r in results) / len(results)
    avg_runtime = sum(r[2].microseconds for r in results) / len(results) / 1000 / 1000
    return random_wins, ab_wins, draws, avg_turns, avg_runtime


def run_alphabeta_games_serial(n_games=100, ab_depth=10):
    results = []
    for _ in tqdm(range(n_games)):
        results.append(do_one_alphabeta_game(verbose=False, depth=ab_depth))

    random_wins = sum(1 for r in results if r[0] == 1)
    ab_wins = sum(1 for r in results if r[0] == 2)
    draws = sum(1 for r in results if r[0] == 0)
    avg_turns = sum(r[1] for r in results) / len(results)
    return random_wins, ab_wins, draws, avg_turns



async def main():
    df_ab = []
    for depth in range(1, 6):
        random_wins, ab_wins, draws, avg_turns, avg_runtime = run_alphabeta_games_parallel(n_games=100, ab_depth=depth)
        df_ab.append(
            {
                "depth": depth,
                "random_wins": random_wins,
                "alphabeta_wins": ab_wins,
                "draws": draws,
                "avg_turns": avg_turns,
                "avg_runtime": avg_runtime,
            }
        )
    df_ab = pd.DataFrame(columns=["depth", "random_wins", "alphabeta_wins", "draws", "avg_turns", "avg_runtime"], data=df_ab)
    print(df_ab)
    df_ab.to_csv("alphabeta_vs_random_diff_depth.csv", index=False)
    
    
    plt.plot(df_ab["depth"], df_ab["alphabeta_wins"], label="Alpha-Beta Wins")
    plt.plot(df_ab["depth"], df_ab["random_wins"], label="Random Wins")
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks(range(0, 101, 10))
    plt.xlabel("Alpha-Beta Search Depth")
    plt.ylabel("Number of Wins")
    plt.title("Alpha-Beta vs Random Player")
    plt.legend()
    plt.grid(True)
    plt.savefig("alphabeta_vs_random.png")
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())


    

    
