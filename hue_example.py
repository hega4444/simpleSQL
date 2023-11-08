from huesdk import Hue #controls for HUE lights
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import os
import sys
from Lexi import Lexi, LexiExternalCommand

lights = {} #global for managing the devices in the house

def set_up_hue():
    global lights

    sys.stdout = open(os.devnull, "w") #deactivate log in stdout


    # Disable the SSL warning if you want to continue without certificate verification
    urllib3.disable_warnings(InsecureRequestWarning)
    # Set up Hue Bridge IP address and User Name
    bridge_ip = "192.168.1.100"
    username = "C9u6yv8ooxuLxnG-1RApypMtaCOCp-37V5Uoqci9"

    # Create a Hue object
    hue = Hue(bridge_ip, username)

    # Get the light with name
    lights['Bathroom light'] = hue.get_light(name="Sofa")
    lights['Ceiling'] = hue.get_light(name="Ceiling")
    lights['Sofa'] = hue.get_light(name="Sofa")
    lights['Iris lamp'] = hue.get_light(name="Iris lamp")
    lights['Hallway'] = hue.get_light(name="Hallway")
    lights['Kitchen'] = hue.get_light(name="Kitchen")

    # Restore the original stdout
    sys.stdout = sys.__stdout__

def turn_on_the_sofa(color = "#FFF000"):
    # KEYS: lights light sofa on turn
    # SUMM: use it to turn on the sofa light and change the color
    # SUMM: enter color in hexa {"color":"#RRGGBB"}
    lights['Sofa'].on()                        
    lights['Sofa'].set_color(hexa=color)
    lights['Sofa'].set_brightness(250)

def turn_off_the_sofa():
    # KEYS: lights light sofa off turn
    # SUMM: use it to turn off the sofa light
    lights['Sofa'].off()

if __name__ == "__main__":
    set_up_hue()

    lexi = Lexi()
    lexi.append_command(LexiExternalCommand(turn_on_the_sofa))
    lexi.append_command(LexiExternalCommand(turn_off_the_sofa))

    lexi.mini_chat_session()
