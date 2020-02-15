#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import time




class UnityComm(object):

  def __init__(self, Agent):
    self.Agent = Agent

    context = zmq.Context()
    self.socket = context.socket(zmq.REP)
    self.socket.bind("tcp://*:5555")
    self.Agent._print2("Unity Comm Connected to 'tcp://*:5555'!")

  def recv_unity(self):
    self.Agent._print2("Waiting for message from Unity...")
    message = self.socket.recv()
    self.Agent._print(f"Unity: {message}")

    return message

  def send_unity(self, message):
    self.Agent._print2(f"Trying to send '{message}' to Unity...")
    try:
      self.socket.send(message)
      self.Agent._print(f"'{message}' send to Unity successfully!")
    except Exception as e:
      self.Agent._print(f"Unable to send '{message}' to Unity...")
      self.Agent._print2(f"Error:\n{e}")


if __name__ == "__main__":
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind("tcp://*:5555")

  while True:
    #  Wait for next request from client

    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'.
    #  Try reducing sleep time to 0.01 to see how blazingly fast it communicates
    #  In the real world usage, you just need to replace time.sleep() with
    #  whatever work you want python to do, maybe a machine learning task?
    time.sleep(1)

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    socket.send(b"World")
