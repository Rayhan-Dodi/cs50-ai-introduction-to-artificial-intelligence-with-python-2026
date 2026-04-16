"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count <= o_count else O


def actions(board):
    return {
        (i, j)
        for i in range(3)
        for j in range(3)
        if board[i][j] is None
    }


def result(board, action):
    i, j = action

    if board[i][j] is not None:
        raise Exception("Invalid move")

    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    lines = []

    lines.extend(board)
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line == [X, X, X]:
            return X
        if line == [O, O, O]:
            return O

    return None


def terminal(board):
    if winner(board) is not None:
        return True
    return all(cell is not None for row in board for cell in row)


def utility(board):
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0


def minimax(board):
    if terminal(board):
        return None

    turn = player(board)

    if turn == X:
        best_score = -math.inf
        best_move = None

        for action in actions(board):
            score = min_value(result(board, action))
            if score > best_score:
                best_score = score
                best_move = action

        return best_move

    else:
        best_score = math.inf
        best_move = None

        for action in actions(board):
            score = max_value(result(board, action))
            if score < best_score:
                best_score = score
                best_move = action

        return best_move


def max_value(board):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
