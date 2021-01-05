from typing import Callable

from reversi import Coordinate, Reversi

def play(reversi: Reversi, player1: Callable[[Reversi, int], Coordinate], player2: Callable[[Reversi, int], Coordinate]) -> int:
    while True:
        # 黑棋下子
        while reversi.next == 1:
            position = player1(reversi, 1)
            status = reversi.place(position, 1)

            if status.startswith('end'):
                return int(status[-1])
            elif status != 'ok':
                raise Exception('Should not reach here!')
        
        # 白棋下子
        while reversi.next == 2:
            position = player2(reversi, 2)
            status = reversi.place(position, 2)

            if status.startswith('end'):
                return int(status[-1])
            elif status != 'ok':
                raise Exception('Should not reach here!')

if __name__ == '__main__':

    import random
    import itertools
    def randomAgent(reversi: Reversi, who: int):
        # assert reversi.next == who
        available = [(y, x) for (y, x) in itertools.product(range(reversi.size), repeat=2) if reversi.good[y][x]]
        return random.choice(available)
    
    from agent import Agent
    agent = Agent()
    
    counter = [0, 0, 0] # [AI赢，随机赢，平局]
    
    import time
    t = time.time()

    for i in range(50):
        print(i, end='\r')
        # AI先手
        reversi = Reversi()
        counter[[2, 0, 1][play(reversi, agent.brain, randomAgent)]] += 1
        # AI后手
        reversi = Reversi()
        counter[[2, 1, 0][play(reversi, randomAgent, agent.brain)]] += 1
    
    print(f'100 episodes finished in {time.time() - t :g} seconds, {counter}')
    