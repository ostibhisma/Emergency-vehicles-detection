import logging 

class Applog:
    def __init__(self,filename):
        self.filename = filename
        self.format = "%(levelname)s -- %(filename)s -- %(asctime)s  -- %(message)s"
        self.level = logging.DEBUG
        self.filemode  = 'w'

    def write(self,obj):
        self.object = obj
        logging.basicConfig(filename= self.filename,format = self.format,
                            level = self.level, filemode=self.filemode)
        self.object = logging.getLogger()
        return self.object