from abc import ABC, abstractmethod
from random import randint
from numpy import argmax


class Player(ABC):

    @abstractmethod
    def next_action(self, state):
        ...


class IndecisivePlayer(Player):

    def next_action(self, state):
        moves = self.next_actions(state)
        return moves[randint(0, len(moves) - 1)]

    @abstractmethod
    def next_actions(self, state):
        ...


class RandomPlayer(IndecisivePlayer):
    def next_actions(self, state):
        return state.actions()


class ConsolePlayer(Player):

    def next_action(self, state):
        actions = state.actions()
        while True:
            try:
                print(state)
                action = input(f'Action [{actions}]: ')
                action = int(action)
                if action in actions:
                    return action
            except Exception:
                pass


class MiniMaxPlayer(IndecisivePlayer):
    def __init__(self, lookahead):
        assert lookahead > 0
        self.lookahead = lookahead

    def next_actions(self, state):
        moves, _ = self.value(state, self.lookahead)
        return moves

    def value(self, state, lookahead):
        if lookahead == 0 or state.gameover():
            return [], 1.0*state.winner()*(lookahead+1)
        behaviour = max if state.player() == 1 else min
        return self.minimax(state, behaviour, lookahead)

    def minimax(self, state, behaviour, lookahead):
        moves, res = [], -10000*state.player()
        for cell in state.actions():
            _, v = self.value(state.move(cell), lookahead-1)
            if res == v:
                moves.append(cell)
            elif behaviour(res, v) == v:
                moves, res = [cell], v
        return moves, res


class NNPlayer(Player):
    def __init__(self, model):
        self.model = model

    def next_action(self, state):
        actions = state.actions()
        current_player = state.player()
        states = [state.move(action).cells for action in actions]
        probs = self.model.predict(states)
        player_probs = [p[current_player] for p in probs]
        return actions[argmax(player_probs)]
