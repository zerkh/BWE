import os, sys
import ConfigParser

class GCWEConfiger():
    '''
    Class for config Global-Context Word Embedding model
    '''
    def __init__(self, filename):
        self.cf_parser = ConfigParser.ConfigParser()
        self.cf_parser.read(filename)
        
        self.word_dim, self.hidden_dim, self.window_size, self.global_hidden_dim\
        = self.cf_parser.getint("parameters", "word_dim"),\
        self.cf_parser.getint("parameters", "hidden_dim"),\
        self.cf_parser.getint("parameters", "window_size"),\
        self.cf_parser.getint("parameters", "global_hidden_dim")