import pkgutil
import os, sys




class Brain(object):

  """
  Instantiates a new Brain object, which cross-references user
  input with a list of modules. Note that the order of brain.modules
  matters, as the Brain will cease execution on the first module
  that accepts a given input.
  """

  def __init__(self, Agent):
    self.Agent = Agent
    self.Mic = self.Agent.Mic
    locations = self.Agent.MODULES_PATH
    if not isinstance(locations, list) and isinstance(locations, str):
      locations = [locations]
    self.modules = self.getModules(locations)

  def getModules(self, locations):
    """
    Dynamically loads all the modules in the modules folder and sorts
    them by the PRIORITY key. If no PRIORITY is defined for a given
    module, a priority of 0 is assumed.
    """
    self.Agent._print(f"\nLooking for modules in: {', '.join(locations)}")
    modules = []
    for finder, name, ispkg in pkgutil.walk_packages(locations):
      try:
        loader = finder.find_module(name)
        mod = loader.load_module(name)
      except Exception as e:
        self.Agent._print2(e)
        self.Agent._print(f"Skipped module `{name}` due to an error.")
      else:
        if hasattr(mod, "isValid"):
          self.Agent._print(f"Found module `{name}`")
          modules.append(mod)
    modules.sort(key=lambda mod: mod.PRIORITY if hasattr(mod, 'PRIORITY')
           else 0, reverse=True)

    return modules

  def query(self, text):
    """
    Passes user input to the appropriate module, testing it against
    each candidate module's isValid function.

    Arguments:
    text -- user input, typically speech, to be parsed by a module
    """
    for module in self.modules:
      if module.isValid(text):
        self.Agent._print(f"'{text}' is a valid phrase for module `{module.__name__}`")
        try:
          module.handle(text, self.Agent.Mic, self.Agent)
        except Exception as e:
          self.Agent._print2(e)
          self.Agent._print('Failed to execute module')
          self.Mic.say("I'm sorry. I had some trouble with that operation. Please try again later.")
        else:
          self.Agent._print(f"Handling of phrase '{text}' by module `{module.__name__}` completed")
        finally:
          return
    self.Agent._print(f"No module was able to handle any of these phrases: {text}")