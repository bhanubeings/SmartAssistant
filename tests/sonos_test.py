from __future__ import print_function, unicode_literals

import os
import sys
import time
import socket
from threading import Thread
from random import choice
from urllib.parse import quote
from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler

import soco
from soco import SoCo




class HttpServer(Thread):
  # A simple HTTP Server in its own thread

  def __init__(self, port):
    super(HttpServer, self).__init__()
    self.daemon = True
    handler = SimpleHTTPRequestHandler
    self.httpd = TCPServer(("", port), handler)

  def run(self):
    # Start the server
    print('Start HTTP server')
    self.httpd.serve_forever()

  def stop(self):
    # Stop the server
    print('Stop HTTP server')
    self.httpd.socket.close()


class SonosUtil(object):
  # Using sonos as the Mic and Speaker

  def __init__(self, name=None, sonos_ip=None, port=6060):
    self.machine_ip = self.detectIPAddress()
    self.port = port

    if name is None and sonos_ip is None:
      raise Exception("either name or sonos_ip must be provided")

    else:
      if name:
        device = soco.discovery.by_name(name)
      elif sonos_ip:
        device = soco.SoCo(sonos_ip)
      elif name and sonos_ip:
        device = soco.SoCo(sonos_ip)
        if device.player_name != name:
          raise Exception("name does not match sonos_ip")

    self.device = device

  def detectIPAddress(self):
    # Return the local IP Address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    machine_ip = s.getsockname()[0]
    s.close()
    return machine_ip

  def play(self, filename):
    # play the file given (wav & mp3 supported)
    print(f"\nPlaying: {filename}")
    netpath = f"http://{self.machine_ip}:{self.port}/{filename}"

    number_in_queue = self.device.play_uri(netpath)



if __name__ == '__main__':
  sonos = SonosUtil("Sonos R")
  server = HttpServer(sonos.port)
  server.start()
  sonos.play("tts_1.mp3")
  duration = sonos.device.get_current_track_info()["duration"]
  print(duration)
  hrs, mins, secs = duration.split(":")
  hrs, mins, secs = int(hrs), int(mins), int(secs)
  total_seconds = hrs*3600 + mins*60 + secs
  print(total_seconds)
  time.sleep(total_seconds)