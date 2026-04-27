
# MYDAILYWORK
# Rule-Based Chatbot (Python)

A simple command-line chatbot built with Python using predefined rules and basic pattern matching.

This project demonstrates:
- Rule-based conversation flow
- Input handling with `if` conditions
- Basic natural language pattern matching using regular expressions
- Fallback responses for unknown queries

## Features

- Greets users (`hi`, `hello`, `hey`)
- Responds to common questions:
  - how are you
  - your name / who are you
  - help/support
  - python learning questions
  - motivation requests
  - thanks
- Exits the chat on:
  - `bye`
  - `goodbye`
  - `exit`
  - `quit`
- Handles unknown input with a default reply

## Project Structure

```text
.
├── rule_based_chatbot.py
└── README.md
```

## Requirements

- Python 3.x

No external libraries are required (uses only Python standard library).

## How to Run

1. Open terminal in the project folder.
2. Run:

```bash
python rule_based_chatbot.py
```

## Example Interaction

```text
Rule-Based Chatbot
Type 'bye' to end the chat.

You: hello
Bot: Hello! How can I help you today?
You: what is your name
Bot: I am a simple rule-based chatbot.
You: thanks
Bot: You're welcome!
You: bye
Bot: Goodbye! Have a nice day.
```

## How It Works

The chatbot uses a `get_response()` function that:
1. Normalizes user input (trim + lowercase)
2. Checks input against predefined rules using regex patterns
3. Returns the first matching response
4. Returns a fallback response if no pattern matches

## Customize the Bot

To add new responses, edit `get_response()` in `rule_based_chatbot.py`:
- Add a new `if re.search(...)` condition
- Return the response you want for that pattern

## Future Improvements

- Move rules to a JSON file for easier updates
- Add conversation memory/context
- Add GUI or web interface
- Add intent scoring for better matching

## License

You can use this project for learning and personal practice.
>>>>>>> a84cb88 (Initial commit: add chatbot project)
