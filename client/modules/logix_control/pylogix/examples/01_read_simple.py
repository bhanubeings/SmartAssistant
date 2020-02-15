'''
the following import is only necessary because eip is not in this directory
'''
import sys
sys.path.append('..')


'''
The simplest example of reading a tag from a PLC

NOTE: You only need to call .Close() after you are done exchanging
data with the PLC.  If you were going to read in a loop or read
more tags, you wouldn't want to call .Close() every time.
'''
from eip import PLC

comm = PLC()
comm.IPAddress = '192.168.1.10'
plc_pass = comm.Read("syswellPass")
print(f"password is {plc_pass}")
comm.Close()
