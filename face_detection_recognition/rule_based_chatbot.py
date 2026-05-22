import re


def get_response(user_input: str) -> str:
    """
    Return a rule-based response for the given user input.
    """
    text = user_input.strip().lower()

    if not text:
        return "Please type something so I can help you."

    # Greeting rules
    if re.search(r"\b(hi|hello|hey)\b", text):
        return "Hello! How can I help you today?"

    # Well-being / small talk rules
    if re.search(r"\b(how are you|how are u)\b", text):
        return "I am doing great. Thanks for asking!"

    if re.search(r"\b(your name|who are you)\b", text):
        return "I am a simple rule-based chatbot."

    # Time/date information rules
    if re.search(r"\b(time)\b", text):
        return "I cannot check live time yet, but you can run your system clock."

    if re.search(r"\b(date|day)\b", text):
        return "I cannot check live date yet, but your device shows it."

    # Help/support rules
    if re.search(r"\b(help|support|assist)\b", text):
        return "Sure. Ask me about greetings, my name, or basic questions."

    # Learning / motivation rules
    if re.search(r"\b(learn python|python)\b", text):
        return "Great choice! Start with variables, loops, functions, and practice daily."

    if re.search(r"\b(motivate|motivation|inspire)\b", text):
        return "You are doing well. Keep learning one small step every day."

    # Thank-you rules
    if re.search(r"\b(thank you|thanks)\b", text):
        return "You're welcome!"

    # Exit rules
    if re.search(r"\b(bye|goodbye|exit|quit)\b", text):
        return "Goodbye! Have a nice day."

    # Fallback rule
    return "Sorry, I did not understand that. Try asking in a different way."


def main() -> None:
    print("Rule-Based Chatbot")
    print("Type 'bye' to end the chat.\n")

    while True:
        user_input = input("You: ")
        bot_response = get_response(user_input)
        print(f"Bot: {bot_response}")

        if re.search(r"\b(bye|goodbye|exit|quit)\b", user_input.strip().lower()):
            break


if __name__ == "__main__":
    main()
