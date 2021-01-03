from PIL import Image, ImageDraw
from reversi import Reversi
import itertools
from typing import List
import numpy as np

WIDTH = 30              # 格子边长
PADDING = 3             # 棋子到外接格子的边距
LINE = 2                # 网格线粗细
BACKGROUND = '#45925D'  # 背景颜色
LINE_COLOR = '#346F47'  # 分割线的颜色
WHITE = '#182818'       # 白棋颜色
BLACK = '#FFFFFF'       # 黑棋颜色
GRAPH_SIZE = 8 * WIDTH

# 画出当前棋盘，输入：棋盘大小，Reversi实例的board；输出图像
def draw_board(board_size, board: List) -> np.ndarray:
    image = Image.new('RGB', (GRAPH_SIZE + 2, GRAPH_SIZE + 2), (BACKGROUND))
    draw = ImageDraw.Draw(image)
    for i in range(0, board_size+2): # 画线
        draw.line((0, i * WIDTH, GRAPH_SIZE - 1, i * WIDTH), fill=LINE_COLOR, width=LINE)
        draw.line((i * WIDTH, 0, i * WIDTH, GRAPH_SIZE - 1), fill=LINE_COLOR, width=LINE)
    for (y, x) in itertools.product(range(board_size), repeat=2):
        if board[y][x]:
            yy, xx = y * WIDTH + LINE, x * WIDTH + LINE
            draw.ellipse((xx + PADDING, yy + PADDING, xx + (WIDTH -1 -PADDING), yy + (WIDTH -1 -PADDING)),
                         fill= WHITE if board[y][x] == 1 else BLACK)
    return np.asarray(image)
