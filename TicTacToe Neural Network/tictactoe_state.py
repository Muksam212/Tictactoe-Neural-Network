from state import make_rules, State

tictactoe_rules = make_rules(cols=3, rows=3, score=3)


class TicTacToeState(State):
    def __init__(self, cells=None):
        super().__init__(cells, cols=3, rows=3, rules=tictactoe_rules)


if __name__ == '__main__':
    from players import ConsolePlayer, MiniMaxPlayer
    from game import play
    states, _ = play(TicTacToeState(), ConsolePlayer(), MiniMaxPlayer(5))
    print(states[-1].state)
