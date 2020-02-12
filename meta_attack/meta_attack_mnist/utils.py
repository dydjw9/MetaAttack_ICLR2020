import os
import sys

class Logger(object):
    def __init__(self, filepath = './log.txt', mode = 'w', stdout = None):
        if stdout == None:
            self.terminal = sys.stdout
        else:
            self.terminal = stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log)
    def flush(self):
        pass

