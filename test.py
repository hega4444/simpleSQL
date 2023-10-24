import readline

# List of available commands (e.g., "play recording", "execute command", etc.)
commands = ["play recording", "execute command", "other command", "play recording a", "play recording casa"]

def custom_completer(text, state):
    text = text.strip()  # Remove leading/trailing whitespace
    # Split the text into words
    words = text.split()
    
    if state == 0:
        # The first word is used for command matching
        word_to_match = words[0]
    else:
        word_to_match = words[-1]  # Match the last word
        
    completions = [cmd for cmd in commands if cmd.startswith(word_to_match)]
    
    if state < len(completions):
        completion = completions[state]
        return completion + ' '
    else:
        return None

readline.set_completer(custom_completer)
readline.parse_and_bind("tab: complete")

while True:
    try:
        command = input("> ")
        print(f"Command entered: {command}")
    except EOFError:
        break
