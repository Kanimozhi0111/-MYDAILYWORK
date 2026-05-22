# Rule-Based Chatbot

A simple command-line chatbot built in Python using regular-expression rules.

## What this project does

- Accepts user input in a loop
- Matches input against predefined intent patterns
- Returns a fixed response for each matched pattern
- Exits cleanly when user types `bye`, `goodbye`, `exit`, or `quit`

## File

- `rule_based_chatbot.py` - chatbot logic and CLI loop

## Requirements

- Python 3.x
- No third-party packages (uses only Python standard library)

## Run

```bash
python rule_based_chatbot.py
```

## Supported intents in current code

- Greetings: `hi`, `hello`, `hey`
- Well-being: `how are you`, `how are u`
- Identity: `your name`, `who are you`
- Time/date queries (returns static guidance)
- Help: `help`, `support`, `assist`
- Python learning: `learn python`, `python`
- Motivation: `motivate`, `motivation`, `inspire`
- Thanks: `thank you`, `thanks`
- Exit: `bye`, `goodbye`, `exit`, `quit`

If no rule matches, it replies with a fallback message.

## Example

```text
Rule-Based Chatbot
Type 'bye' to end the chat.

You: hello
Bot: Hello! How can I help you today?
You: motivate me
Bot: You are doing well. Keep learning one small step every day.
You: bye
Bot: Goodbye! Have a nice day.
```

## Customize rules

Edit `get_response()` in `rule_based_chatbot.py` and add new `if re.search(...)` conditions above the fallback return.
