import os,sys,datetime
import time

LOG_FILE = ".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def Enable_Logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w",encoding="utf-8") as fp:
            pass
    sys.stdout = Logger(LOG_FILE, sys.stdout)
    pass

def timer_logger(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args,**kwargs)
        print(f"运行时间: {time.perf_counter() - start}s")
        return result
    return wrapper