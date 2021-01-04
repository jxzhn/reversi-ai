from typing import Tuple

import itertools

Coordinate = Tuple[int, int]
SIZE = 8

class Reversi():
    def __init__(self):
        self.size = SIZE
        self.board = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

        # 初始时棋盘上有四颗棋子
        ii = SIZE // 2
        self.board[ii - 1][ii] = 1
        self.board[ii][ii - 1] = 1
        self.board[ii - 1][ii - 1] = 2
        self.board[ii][ii] = 2

        self.number = {1: 2, 2: 2} # 各棋子个数
        
        self.next = 1 # 轮到哪个颜色
        self.good = [[False for _ in range(SIZE)] for _ in range(SIZE)]
        self.analyse()

        self.recent = (-1, -1) # 记录最近一个棋子的位置（用于GUI提示）
    
    def place(self, postion: Coordinate, player: int) -> str:
        y, x = postion

        if player != self.next or y < 0 or y >= self.size or x < 0 or x >= self.size or not self.good[y][x]:
            return 'no'
        
        a = player  # 己方编号
        b = 2 if a == 1 else 1 # 敌方编号

        self.board[y][x] = a
        self.recent = (y, x)
        self.number[a] += 1

        # 进行颜色翻转
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            yy, xx = y + dy, x + dx
            if yy < 0 or yy >= self.size or xx < 0 or xx >= self.size or self.board[yy][xx] != b:
                continue

            yy, xx = yy + dy, xx + dx
            while 0 <= yy and yy < self.size and 0 <= xx and xx < self.size:
                if self.board[yy][xx] == a:
                    # 将(y, x)到(yy, xx)之前的棋全部翻转颜色
                    yyy, xxx = y + dy, x + dx
                    while (yyy, xxx) != (yy, xx):
                        self.board[yyy][xxx] = a
                        self.number[a] += 1
                        self.number[b] -= 1
                        yyy, xxx = yyy + dy, xxx + dx
                    break
                elif not self.board[yy][xx]:
                    break
                yy, xx = yy + dy, xx + dx
        
        # number[a]一定不是0，不需要判断
        if self.number[b] == 0:
            # 一方全部棋子被翻转，则另一方获胜
            self.next = 0
            return f'end {a}'
        elif self.number[a] + self.number[b] == self.size * self.size:
            # 棋盘已下满，进行结算
            self.next = 0
            return f'end {a}' if self.number[a] > self.number[b] else\
                ('end 0' if self.number[a] == self.number[b] else f'end {b}')

        self.next = 2 if player == 1 else 1
        # 分析可下棋位置
        self.analyse()

        if self.next == 0:
            # 说明双方均无位置可以下，进行结算
            return f'end {a}' if self.number[a] > self.number[b] else\
                ('end 0' if self.number[a] == self.number[b] else f'end {b}')

        return 'ok'
    
    # 分析可以下棋的位置
    def analyse(self, reEnter: bool = False):
        a = self.next   # 己方编号
        b = 2 if a == 1 else 1 # 敌方编号

        movable = False # 是否至少有一个可以下的位置

        for y, x in itertools.product(range(self.size), repeat=2):
            if self.board[y][x]:
                self.good[y][x] = False
                continue

            marked = False

            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                yy, xx = y + dy, x + dx
                if yy < 0 or yy >= self.size or xx < 0 or xx >= self.size or self.board[yy][xx] != b:
                    continue

                yy, xx = yy + dy, xx + dx
                while 0 <= yy and yy < self.size and 0 <= xx and xx < self.size:
                    if self.board[yy][xx] == a:
                        self.good[y][x] = True
                        marked = True
                        movable = True
                        break
                    elif not self.board[yy][xx]:
                        break
                    yy, xx = yy + dy, xx + dx
                if marked:
                    break
            
            if not marked:
                self.good[y][x] = False
        
        if not movable:
            if not reEnter:
                # 一个可以下的位置都没有，轮到另一方下棋
                self.next = b
                self.analyse(True)
            else:
                # 双方都没有位置下棋，准备进行结算
                self.next = 0