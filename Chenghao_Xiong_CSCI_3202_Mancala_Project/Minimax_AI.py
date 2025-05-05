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
            # print("Invalid move\n")
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
        
        
class MinimaxPlayer:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.game_tree = {}  # 用于存储游戏树
        self.node_counter = 0  # 用于给节点分配唯一ID

    @staticmethod
    def utility(game: Mancala):
        """计算效用函数：Max玩家的Mancala中的石头数 - Min玩家的Mancala中的石头数"""
        # 假设玩家0是Max，玩家1是Min
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
        """检查游戏是否结束"""
        # 检查玩家0的所有坑是否为空
        player0_empty = all(game.board[i] == 0 for i in range(game.pits_per_player))
        # 检查玩家1的所有坑是否为空
        player1_empty = all(game.board[i] == 0 for i in range(game.pits_per_player, 2 * game.pits_per_player))

        return player0_empty or player1_empty

    def get_next_move(self, game):
        """
        使用Minimax算法确定最佳移动

        Args:
            game: Mancala游戏实例

        Returns:
            int: 最优的pit索引
        """
        # 重置游戏树和节点计数器
        self.game_tree = {}
        self.node_counter = 0

        # 创建根节点
        root_id = self.node_counter
        self.node_counter += 1
        self.game_tree[root_id] = {"state": copy.deepcopy(game), "children": [], "value": None, "move": None, "depth": 0}

        # 运行minimax算法
        _, best_move = self.minimax(root_id, 0, True)

        return best_move

    def minimax(self, node_id, depth, is_maximizing):
        """
        Minimax算法实现

        Args:
            node_id: 当前节点ID
            depth: 当前深度
            is_maximizing: 是否是Max玩家的回合

        Returns:
            tuple: (最佳值, 最佳移动)
        """
        node = self.game_tree[node_id]
        game: Mancala = node["state"]

        # 检查是否为终止节点
        if depth == self.max_depth or self.is_game_over(game):
            value = self.utility(game)
            node["value"] = value
            return value, None

        # 获取当前玩家的有效移动
        valid_moves = []
        player = 0 if is_maximizing else 1
        start_pit = 0 if player == 0 else game.pits_per_player

        for i in range(start_pit, start_pit + game.pits_per_player):
            if game.board[i] > 0:
                valid_moves.append(i)

        if not valid_moves:
            # 如果没有有效移动，返回当前状态的效用值
            value = self.utility(game)
            node["value"] = value
            return value, None

        # 初始化最佳值和最佳移动
        best_value = float("-inf") if is_maximizing else float("inf")
        best_move = None

        # 对每个有效移动进行评估
        for move in valid_moves:
            # 创建游戏的副本并执行移动
            game_copy = copy.deepcopy(game)

            # 确定当前玩家
            current_player = 0 if is_maximizing else 1

            # 执行移动并获取下一个玩家
            next_player = game_copy.play(move)

            # 确定下一个玩家是否与当前玩家相同
            next_player_is_current = next_player == current_player

            # 创建子节点
            child_id = self.node_counter
            self.node_counter += 1

            self.game_tree[child_id] = {"state": game_copy, "children": [], "value": None, "move": move, "depth": depth + 1}

            # 将子节点添加到当前节点的子节点列表中
            node["children"].append(child_id)

            # 递归调用minimax
            # 如果下一个玩家与当前玩家相同，则保持is_maximizing不变
            next_is_maximizing = is_maximizing if next_player_is_current else not is_maximizing
            value, _ = self.minimax(child_id, depth + 1, next_is_maximizing)

            # 更新最佳值和最佳移动
            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        # 更新节点的值
        node["value"] = best_value

        return best_value, best_move


def do_one_minimax_game(verbose=False, minimax_depth=5):
    """
    进行一局游戏，玩家1使用随机策略，玩家2使用Minimax策略

    Args:
        verbose: 是否显示游戏过程
        minimax_depth: Minimax算法的最大深度

    Returns:
        int: 胜利者，1为玩家1，2为Minimax玩家，0为平局
        int: 胜利者所用的回合数
    """
    start_time = datetime.now()
    game = Mancala(pits_per_player=6, stones_per_pit=4)
    minimax_player = MinimaxPlayer(max_depth=minimax_depth)

    if verbose:
        game.display_board()

    while True:
        # Player 1 执行随机动作
        move = game.random_move_generator()
        game.play(move)
        if verbose:
            game.display_board()
        if game.winning_eval() is not None:
            break

        # Player 2 执行Minimax动作
        minimax_move = minimax_player.get_next_move(game)
        if minimax_move is not None:
            game.play(minimax_move)
            if verbose:
                game.display_board()
            if game.winning_eval() is not None:
                break

    winner = game.winning_eval()
    end_time = datetime.now()
    return winner, game.turns["p2" if winner == 2 else "p1"], end_time - start_time



def run_games_parallel(n_games=100, minimax_depth=3):
    """并行执行多个游戏"""
    # 创建进程池
    with multiprocessing.Pool() as pool:
        # 创建任务列表
        results_async = []
        for _ in range(n_games):
            # 提交游戏任务到进程池
            result_async = pool.apply_async(do_one_minimax_game, kwds={"verbose": False, "minimax_depth": minimax_depth})
            results_async.append(result_async)

        # 使用tqdm显示进度
        results = []
        for result_async in tqdm(results_async, total=len(results_async)):
            results.append(result_async.get())  # 获取结果

    # 统计结果
    random_wins = 0
    minimax_wins = 0
    draws = 0

    for winner, _ in results:
        if winner == 1:
            random_wins += 1
        elif winner == 2:
            minimax_wins += 1
        else:
            draws += 1

    return random_wins, minimax_wins, draws, results


# 测试Minimax vs 随机策略
def run_minimax_games_parallel(n_games=100, minimax_depth=5):
    avg_runtime = []
    with multiprocessing.Pool() as pool:
        results_async = [pool.apply_async(do_one_minimax_game, kwds={"verbose": False, "minimax_depth": minimax_depth}) for _ in range(n_games)]
        results = [r.get() for r in tqdm(results_async)]
    random_wins = sum(1 for r in results if r[0] == 1)
    minimax_wins = sum(1 for r in results if r[0] == 2)
    draws = sum(1 for r in results if r[0] == 0)
    avg_turns = sum(r[1] for r in results) / len(results)
    avg_runtime = sum(r[2].microseconds for r in results) / len(results) / 1000 / 1000
    return random_wins, minimax_wins, draws, avg_turns, avg_runtime

def run_minimax_games_serial(n_games=100, minimax_depth=5):
    results = []
    for _ in tqdm(range(n_games)):
        result = do_one_minimax_game(verbose=False, minimax_depth=minimax_depth)
        results.append(result)

    random_wins = sum(1 for r in results if r[0] == 1)
    minimax_wins = sum(1 for r in results if r[0] == 2)
    draws = sum(1 for r in results if r[0] == 0)
    avg_turns = sum(r[1] for r in results) / len(results)

    return random_wins, minimax_wins, draws, avg_turns


# 执行主函数
async def main():
    # Exp4: 随机策略 vs Minimax策略, 对比minimax_depth=1,2,3,4,5
    df4 = []
    for minimax_depth in range(1, 6):
        random_wins, minimax_wins, draws, avg_turns, avg_runtime = run_minimax_games_parallel(n_games=100, minimax_depth=minimax_depth)
        df4.append(
            {
                "minimax_depth": minimax_depth,
                "random_wins": random_wins,
                "minimax_wins": minimax_wins,
                "draws": draws,
                "avg_turns": avg_turns,
                "avg_runtime": avg_runtime,
            }
        )
    df4 = pd.DataFrame(columns=["minimax_depth", "random_wins", "minimax_wins", "draws", "avg_turns", "avg_runtime"], data=df4)
    print(df4)
    df4.to_csv("minimax_vs_random_diff_depth.csv", index=False)
    
    plt.plot(df4["minimax_depth"], df4["minimax_wins"], label="Minimax Wins")
    plt.plot(df4["minimax_depth"], df4["random_wins"], label="Random Wins")
    plt.xticks([1, 2, 3, 4, 5])  # 离散整数表示搜索深度
    plt.yticks(range(0, 101, 10))  # 胜局数 0 到 100，步长为 10
    plt.xlabel("Minimax Search Depth")
    plt.ylabel("Number of Wins")
    plt.title("Minimax vs Random Player")
    plt.legend()
    plt.grid(True)
    plt.savefig("minimax_vs_random.png")
    plt.show()



if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
    
    
    

    
