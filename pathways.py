import os, sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.getcwd())




# Agent main directory
APP_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
APP_PATH = os.path.join(APP_PATH, "SmartAssistant")
GUI_PATH = os.path.join(APP_PATH, "GUI")

# general pathways
TEMP_PATH = os.path.join(APP_PATH, "temp")
CLIENT_PATH = os.path.join(APP_PATH, "client")
MODULES_PATH = os.path.join(CLIENT_PATH, "modules")

# sub modules pathways
CONTROLLER_PATH = os.path.join(MODULES_PATH, "logix_control")
DL_PATH = os.path.join(MODULES_PATH, "deep_learning")

# sub^2 modules pathways
CHATBOT_PATH = os.path.join(DL_PATH, "chatbot")
LOGIX_PATH = os.path.join(CONTROLLER_PATH, "pylogix")

if __name__ == '__main__':
  print(APP_PATH)
  print(GUI_PATH)
  print(TEMP_PATH)
  print(CLIENT_PATH)
  print(MODULES_PATH)
  print(CONTROLLER_PATH)
  print(DL_PATH)
  print(LOGIX_PATH)
  print(CHATBOT_PATH)
