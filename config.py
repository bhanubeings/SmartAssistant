# loads configuration for smart assistant
import yaml




with open("config.yml") as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

# config_dict = {
#   "name": "Nyx",
#   "version": 1.0,

#   "controller_ip": "192.168.1.10",
#   "port": 6060,

#   "user_firstname": "Nixon",
#   "user_lastname": "Cheng",

#   "hotword_train_bypass": True,
#   "hotword_train_amount": 10
# }

config_dict["AFFIRM"] = ["Alright.",
                         "OK!",
                         f"As you wish, {config_dict['user_firstname']}!",
                         "Right away!",
                         "As you wish.",
                         "So be it."]

config_dict["REJECT"] = ["Sorry, it's over my limit.",
                         "I can't.",
                         "It's impossible.",
                         "I am unable to do it!",
                         "I can't complete that task."]

config_dict["GREET"] = [f"Yes, {config_dict['user_firstname']}?",
                        "What's up?",
                        "Yes?",
                        "Anything?"]

if __name__ == '__main__':
  # for c in config_dict:
  #   print(f"{c}: {config_dict[c]}")
  print(config_dict)

  # with open("config.yml") as f:
  #   data = yaml.load(f, Loader=yaml.FullLoader)
  #   print(data)
