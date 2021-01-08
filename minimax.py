from typing import Tuple

import itertools
from reversi import Reversi, Coordinate
from copy import deepcopy

evaluate_matrix = [[100, -20, 50, 25, 25, 50, -20, 100],
                   [-20, 80, 25, 10, 10, 25, 80, -20],
                   [50, 25, 20, 5, 5, 20, 25, 50],
                   [25, 10, 5, 0, 0, 5, 10, 25],
                   [25, 10, 5, 0, 0, 5, 10, 25],
                   [50, 25, 20, 5, 5, 20, 25, 50],
                   [-20, 80, 25, 10, 10, 25, 80, -20],
                   [100, -20, 50, 25, 25, 50, -20, 100]]

def evaluate(reversi: Reversi) -> int:
    total = 0
    for y, x in itertools.product(range(reversi.size), repeat=2):
        if reversi.board[y][x]:
            total += evaluate_matrix[y][x] if reversi.board[y][x] == 1 else -evaluate_matrix[y][x]
    return total

DEPTH = 4

class Agent:
    @staticmethod
    def brain(reversi: Reversi, who: int) -> Coordinate:
        position, _ = Agent.search(reversi, who, -10000000, 10000000, DEPTH)
        return position
    
    @staticmethod
    def search(reversi: Reversi, who: int, alpha: int, beta: int, depth: int) -> Tuple[Coordinate, int]:
        if depth <= 0 or reversi.next == 0:
            return None, evaluate(reversi)

        if reversi.next != who:
            return Agent.search(reversi, 2 if who == 1 else 2, alpha, beta, depth-1)

        available = [(y, x) for y, x in itertools.product(range(reversi.size), repeat=2) if reversi.good[y][x]]
        best = None

        if who == 1: # Max
            for position in available:
                reversi_ = deepcopy(reversi)
                reversi_.place(position, 1)
                _, score = Agent.search(reversi_, 2, alpha, beta, depth-1)

                if score > alpha: # alpha就是Max能获得的最大效益值
                    alpha = score
                    best = position
                    if alpha >= beta:
                        return None, alpha

            return best, alpha
        else: # Min
            for position in available:
                reversi_ = deepcopy(reversi)
                reversi_.place(position, 2)
                _, score = Agent.search(reversi_, 1, alpha, beta, depth-1)

                if score < beta: # beta就是Min能获得的最小效益值
                    beta = score
                    best = position
                    if beta <= alpha:
                        return None, beta

            return best, beta