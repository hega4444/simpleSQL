from Lexi import Lexi

if __name__ == "__main__":

    lexi = Lexi()

    user_text = input("You:")
    while user_text != "exit":

        lexi.new_user_input(user_text)
        lexi.process_inbound()

        user_text = input("You:")