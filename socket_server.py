import socket
import time




class Socket(object):

  def __init__(self, Agent):
    self.Agent = Agent

    self.listen_socket = socket.socket()
    self.socket_port = self.Agent.socket_port
    self.max_conn = self.Agent.socket_max_conn
    self.socketIP = socket.gethostname()

    self.listen_socket.bind((self.socketIP, self.socket_port))
    self.listen_socket.listen(self.max_conn)
    self.Agent._print(f"Server started at {self.socketIP} on port {self.socket_port}")
    self.client_socket, self.address = self.listen_socket.accept()
    self.Agent._print(f"New connection made!")

  def recv_msg(self):
    msg = self.client_socket.recv(1024).decode()
    self.Agent._print2(f"Recieved message: {msg}")
    return msg

  def send_msg(self, msg):
    self.Agent._print2(f"Sending message: {msg}")
    self.client_socket.send(msg.encode("ASCII"))

