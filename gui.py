from typing import Callable

import tkinter
import tkinter.messagebox
import itertools
from reversi import Coordinate, Reversi
from agent import Agent

WIDTH = 30      # 格子边长
PADDING = 3     # 棋子到外接格子的边距
LINE = 2        # 网格线粗细

class ReversiGUIManager():
    def __init__(self, size: int = 8):
        window = tkinter.Tk()
        window.title('Reversi')

        window.geometry('{}x{}'.format(size*WIDTH, size*WIDTH))
        window.resizable(0, 0)

        canvas = tkinter.Canvas(window, width=size*WIDTH, height=size*WIDTH, background='#45925D')
        canvas.pack()

        for i in range(0, size + 1):
            canvas.create_line(0, i*WIDTH, size*WIDTH - 1, i*WIDTH, width=LINE, fill='#346F47')
            canvas.create_line(i*WIDTH, 0, i*WIDTH, size*WIDTH - 1, width=LINE, fill='#346F47')

        self.window = window
        self.canvas = canvas
        self.size = size
        self.colors = {1: '#182818', 2: '#FFFFFF'}
    
    def play(self, reversi: Reversi, who: int, brain: Callable[[Reversi, int], Coordinate]):
        ai = 2 if who == 1 else 1
        
        # 辅助函数：绘制最近一个棋子的虚线框，并重绘整个棋盘上的棋子（懒得判断翻转）
        def draw():
            if reversi.recent:
                self.canvas.delete('hint-recent')
                y, x = reversi.recent
                yy, xx = y * WIDTH, x * WIDTH 
                self.canvas.create_rectangle(xx, yy, xx + (WIDTH-1), yy + (WIDTH-1), fill='', 
                    outline=self.colors[reversi.board[y][x]], width=LINE, tag='hint-recent')

            for (y, x) in itertools.product(range(self.size), repeat=2):
                if reversi.board[y][x]:
                    self.canvas.delete(f'chess-{y}-{x}')
                    yy, xx = y * WIDTH, x * WIDTH
                    self.canvas.create_oval(xx + PADDING, yy + PADDING, xx + (WIDTH-1-PADDING), yy + (WIDTH-1-PADDING), 
                        fill=self.colors[reversi.board[y][x]], tag=f'chess-{y}-{x}')
            self.window.update()

        # 鼠标单击事件的绑定函数，整个GUI游戏过程靠鼠标单击事件驱动
        def turn(event: tkinter.Event):
            y, x = event.y// WIDTH, event.x // WIDTH
            if y < 0 or y >= self.size or x < 0 or x > self.size:
                return
            
            status = reversi.place((y, x), who)

            if status == 'no':
                return

            self.canvas.delete('mouse-hint') # 清除鼠标位置提示
            draw()
            print('You place at {}, black: {}, white: {}'.format((x, y), reversi.number[1], reversi.number[2]))
            
            if status.startswith('end'):
                msg = 'Draw' if status == 'end 0' else f'Player {status[-1]} wins!'
                tkinter.messagebox.showinfo(title='Game Over', message=msg)
                self.canvas.unbind('<Motion>')
                self.canvas.unbind('<Button-1>')
                return
            
            # 轮到AI下棋

            aiPlayed = False # 记录AI是否已经下过一次（再下说明刚刚玩家没位置下），以判断是否打印提示

            while reversi.next == ai:
                if aiPlayed:
                    print('You have no position to place your chess, so it\'s still AI\'s turn.')

                position = brain(reversi, ai)
                status = reversi.place(position, ai)

                if status == 'no':
                    raise Exception('FUCK! THIS AI IS SHIT!')

                print('AI place at {}, black: {}, white: {}'.format(
                    (position[1], position[0]), reversi.number[1], reversi.number[2]
                ) )
                draw()

                if status.startswith('end'):
                    msg = 'Draw' if status == 'end 0' else f'Player {status[-1]} wins!'
                    tkinter.messagebox.showinfo(title='Game Over', message=msg)
                    self.canvas.unbind('<Motion>')
                    self.canvas.unbind('<Button-1>')
                    return
                
                aiPlayed = True
            
            if not aiPlayed:
                print('AI have no position to place chess, it\'s still your turn.')
        
        # 需要一个辅助函数来绑定鼠标移动事件，来提示鼠标位置
        def hint(event: tkinter.Event):
            self.canvas.delete('mouse-hint')

            y, x = event.y// WIDTH, event.x // WIDTH
            if y < 0 or y >= self.size or x < 0 or x > self.size or not reversi.good[y][x]: # assert reversi.next == who
                return

            yy, xx = y * WIDTH, x * WIDTH
            self.canvas.create_rectangle(xx, yy, xx + (WIDTH-1), yy + (WIDTH-1),fill='#A2C8AE', outline='', tag='mouse-hint')
        
        draw() # 绘制初始棋局

        if ai == 1: # AI 先手
            position = brain(reversi, ai)
            _ = reversi.place(position, ai)

            draw()
            print('AI place at {}, black: {}, white: {}'.format(
                    (position[1], position[0]), reversi.number[1], reversi.number[2]
            ) )

        self.canvas.bind('<Motion>', hint)
        self.canvas.bind('<Button-1>', turn)

        self.window.mainloop()

if __name__ == '__main__':
    reversi = Reversi()

    # import random
    # def randomAgent(reversi: Reversi, who: int):
    #     # assert reversi.next == who
    #     available = [(y, x) for (y, x) in itertools.product(range(reversi.size), repeat=2) if reversi.good[y][x]]
    #     return random.choice(available)
    agent = Agent()
    
    gui = ReversiGUIManager()
    who = 1 if tkinter.messagebox.askquestion(title='Reversi', message='user play first?') == 'yes' else 2
    gui.play(reversi, who, agent.brain)