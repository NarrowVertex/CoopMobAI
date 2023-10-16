import time

class Debug:

    @classmethod
    def log(cls, message):
        print(f'[{time.strftime("%H:%M:%S", time.localtime())}] {message}')
