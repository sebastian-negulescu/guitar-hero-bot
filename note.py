from enum import Enum


class NoteType(Enum):
    regular = 0
    base = 1
    

class Note:
    def __init__(self, template, colour, shape):
        self.template = template
        self.colour = colour
        self.shape = shape
        self.bounds = None

