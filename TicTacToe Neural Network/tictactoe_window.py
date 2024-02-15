from pygame.gfxdraw import aacircle
from pygame.draw import line
from tictactoe_state import TicTacToeState
from board_window import BoardWindow

CELL_COLORS = [(255, 255, 255),  # Empty cell
               (180, 40, 30),    # Player 1
               (180, 180, 30)]   # Player -1
BOARD_COLOR = (20, 30, 10)
GRID_COLOR = (200, 200, 200)


class TicTacToeWindow(BoardWindow):

    def __init__(self, cols=3, rows=3, state=None, autoplayer=None):
        super().__init__(
            title='Tic Tac Toe',
            state=state or TicTacToeState(),
            autoplayer=autoplayer,
            cols=cols, rows=rows,
            grid_size=70, cell_padding=9,
            padding_v=0, padding_h=0)

    def action_for_cell(self, pos):
        return pos.col + pos.row*self.cols

    def draw_background(self, screen):
        screen.fill(BOARD_COLOR)
        gs = self.grid_size
        for i in range(1, self.cols):
            line(screen, GRID_COLOR, (i*gs, 0), (i*gs, self.rows*gs))
        for j in range(1, self.rows):
            line(screen, GRID_COLOR, (0, j*gs), (self.cols*gs, j*gs))

    def draw_cell(self, screen, player, rect):
        color = CELL_COLORS[player]
        if player == 1:
            line(screen, color,
                 (rect.left, rect.top),
                 (rect.left + rect.width, rect.top + rect.height),
                 width=3)
            line(screen, color,
                 (rect.left + rect.width, rect.top),
                 (rect.left, rect.top + rect.height),
                 width=3)
        elif player == -1:
            radius = int(rect.width/2)
            center = rect.left + radius, rect.top + radius
            # pygame.draw.circle(self.screen, color, center, radius, width=3)
            aacircle(screen, *center, radius, color)
            aacircle(screen, *center, radius-1, color)
            aacircle(screen, *center, radius-2, color)


if __name__ == '__main__':
    window = TicTacToeWindow()
    window.show()
    print('hello')
