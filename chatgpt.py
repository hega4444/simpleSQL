import os
import openai

api_key = os.environ.get("MY_API_KEY")
if api_key:
    print("API key found!")
    openai.api_key = api_key
else:
    print("API key not found.")

def interactive_chat():
    print("Starting interactive chat with GPT-3.5 Turbo...")
    print("Type 'exit' to end the session.\n")
    
    conversation = {
        'messages': [{'role': 'system', 'content': """you are exentric and always lying to me """}]
    }

    while True:
        # Get user input
        user_input = input("You: ")

        # Check for exit command
        if user_input.lower() == "exit":
            break

        # Update the conversation messages
        conversation['messages'].append({'role': 'user', 'content': user_input})

        # Use the API to get a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation['messages']
        )

        assistant_message = response['choices'][0]['message']['content']
        print("GPT-3.5 Turbo:", assistant_message)

        # Append the assistant's message to the conversation
        conversation['messages'].append({'role': 'assistant', 'content': assistant_message})

        
    print(conversation)
interactive_chat()
