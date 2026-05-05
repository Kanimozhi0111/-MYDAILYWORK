"""Unbeatable Tic-Tac-Toe using Minimax with alpha-beta pruning."""

from __future__ import annotations

from typing import List, Optional, Tuple


Board = List[str]
WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def print_board(board: Board) -> None:
    """Print board as a 3x3 grid."""
    display = [cell if cell != " " else str(i + 1) for i, cell in enumerate(board)]
    print()
    print(f" {display[0]} | {display[1]} | {display[2]} ")
    print("---+---+---")
    print(f" {display[3]} | {display[4]} | {display[5]} ")
    print("---+---+---")
    print(f" {display[6]} | {display[7]} | {display[8]} ")
    print()


def get_winner(board: Board) -> Optional[str]:
    """Return 'X' or 'O' if someone has won, else None."""
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: Board) -> bool:
    return " " not in board and get_winner(board) is None


def available_moves(board: Board) -> List[int]:
    return [i for i, cell in enumerate(board) if cell == " "]


def minimax(
    board: Board,
    depth: int,
    is_maximizing: bool,
    ai_mark: str,
    human_mark: str,
    alpha: int,
    beta: int,
) -> int:
    """Evaluate board from AI perspective."""
    winner = get_winner(board)
    if winner == ai_mark:
        return 10 - depth
    if winner == human_mark:
        return depth - 10
    if is_draw(board):
        return 0

    if is_maximizing:
        best_score = -10_000
        for move in available_moves(board):
            board[move] = ai_mark
            score = minimax(
                board, depth + 1, False, ai_mark, human_mark, alpha, beta
            )
            board[move] = " "
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score

    best_score = 10_000
    for move in available_moves(board):
        board[move] = human_mark
        score = minimax(board, depth + 1, True, ai_mark, human_mark, alpha, beta)
        board[move] = " "
        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return best_score


def best_ai_move(board: Board, ai_mark: str, human_mark: str) -> int:
    """Choose best move for AI using Minimax."""
    best_score = -10_000
    move_choice = -1
    for move in available_moves(board):
        board[move] = ai_mark
        score = minimax(
            board=board,
            depth=0,
            is_maximizing=False,
            ai_mark=ai_mark,
            human_mark=human_mark,
            alpha=-10_000,
            beta=10_000,
        )
        board[move] = " "
        if score > best_score:
            best_score = score
            move_choice = move
    return move_choice


def get_human_move(board: Board) -> int:
    while True:
        raw = input("Choose your move (1-9): ").strip()
        if not raw.isdigit():
            print("Please enter a number from 1 to 9.")
            continue
        position = int(raw)
        if position < 1 or position > 9:
            print("Move must be between 1 and 9.")
            continue
        move = position - 1
        if board[move] != " ":
            print("That spot is already taken. Try another one.")
            continue
        return move


def choose_mark() -> Tuple[str, str]:
    while True:
        choice = input("Do you want to be X or O? ").strip().upper()
        if choice in {"X", "O"}:
            human_mark = choice
            ai_mark = "O" if human_mark == "X" else "X"
            return human_mark, ai_mark
        print("Invalid choice. Please type X or O.")


def play_game() -> None:
    print("Tic-Tac-Toe: Human vs Unbeatable AI")
    print("Enter positions as numbers 1-9.")
    board = [" "] * 9
    human_mark, ai_mark = choose_mark()

    current = "X"
    while True:
        print_board(board)

        if current == human_mark:
            move = get_human_move(board)
            board[move] = human_mark
        else:
            ai_move = best_ai_move(board, ai_mark, human_mark)
            board[ai_move] = ai_mark
            print(f"AI chose position {ai_move + 1}.")

        winner = get_winner(board)
        if winner is not None:
            print_board(board)
            if winner == human_mark:
                print("You win! (This should only happen if rules are modified.)")
            else:
                print("AI wins!")
            break
        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        current = "O" if current == "X" else "X"


def main() -> None:
    while True:
        play_game()
        again = input("Play again? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
