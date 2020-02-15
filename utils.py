import requests




class Util(object):

  def __init__(self, verbose):
    self.verbose = verbose

  @classmethod
  def checkInternet(self):
    url = 'http://www.google.com/'
    timeout = 5
    try:
      _ = requests.get(url, timeout=timeout)
      return True
    except requests.ConnectionError:
      return  False

  def _print(self, content):
    if self.verbose >= 1:
      print(content)

  def _print2(self, content):
    if self.verbose >= 2:
      print(content)

