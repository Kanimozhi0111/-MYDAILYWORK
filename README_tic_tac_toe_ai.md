# Tic-Tac-Toe AI (Unbeatable Minimax)

A command-line Tic-Tac-Toe game where a human plays against an AI that uses Minimax with alpha-beta pruning.

## What this project does

- Lets the player choose `X` or `O`
- Uses perfect-play search (Minimax) for AI decisions
- Prunes the search tree with alpha-beta bounds for efficiency
- Validates user input (`1-9`, empty spot only)
- Supports replay (`Play again? (y/n)`)

## File

- `tic_tac_toe_ai.py` - complete game logic, AI, and CLI loop

## Requirements

- Python 3.8+ (or any Python 3 version with typing support used here)
- No external dependencies

## Run

```bash
python tic_tac_toe_ai.py
```

## How to play

1. Choose your mark: `X` or `O`.
2. Enter a move by typing a number from `1` to `9`.
3. Board numbering:

```text
 1 | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9
```

4. After each move, the game checks win/draw state.

## AI details

- `best_ai_move()` evaluates every available move.
- `minimax()` scores positions:
  - AI win: `10 - depth`
  - Human win: `depth - 10`
  - Draw: `0`
- Alpha-beta pruning stops exploring branches that cannot improve the result.

With current rules, the AI should never lose.
