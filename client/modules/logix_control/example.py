import os, sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.getcwd())

import speech_recognition as sr
import pyaudio
from gtts import gTTS
# from reference import IP
import pygame
import time


import pylogix
from pylogix.eip import PLC



class Bot(object):
  def __init__(self, ip):

    # Initialize the PLC settings (IP address)
    self.ip = ip
    self.PLC = PLC()
    self.PLC.IPAddress = ip

    # Initialize the PLC tags in a dictionary for further usage
    self.tag_dict = {
    'CK_light': 'DL11.output',
    'YH_light': 'DL12.output',
    'YX_light': 'DL13.output',
    'Entrance': 'DL14.output'
    }

    self.count = 0

    self.loop()

  def stt(self):
    r = sr.Recognizer()
    m = sr.Microphone()

    with sr.Microphone() as source:
      print("Say something!")
      audio = r.listen(source)

    try:
      text = r.recognize_google(audio)
      print(f'You: {text}')
      return text
    except sr.UnknownValueError:
      print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
      print(f"Could not request results from Google Speech Recognition service; {e}")

  def tts(self, input_text):
    print(f'Bot: {input_text}')
    mp3_file = rf'D:\Nyx\Codes\smart_assistant\tts{self.count%2}.mp3'
    tts = gTTS(input_text)
    tts.save(mp3_file)

    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()
    self.count += 1
    time.sleep(2)

  def run(self):
    text = self.stt()

    if text == 'switch on the light':
      self.tts('Alright! Switching on the light')
      self.PLC.Write(self.tag_dict['Entrance'], True)

    elif text == 'switch off the light':
      self.tts('Alright! Switching off the light')
      self.PLC.Write(self.tag_dict['Entrance'], False)

    else:
      self.tts('Your command is invalid')

  def loop(self):
    self.PLC.Write(self.tag_dict['Entrance'], True)
    # while True:
    #   self.run()


if __name__ == '__main__':
  Bot(ip='192.168.1.10')

