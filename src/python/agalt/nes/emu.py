from agalt.emulator import Emulator
import fceu

class FCEU(Emulator):

    def __init__(self):
        Emulator.__init__(self)

    def load_game(self, filename):
        fceu.load_game(filename)

    def emulate_frame(self, action):
        return fceu.emulate_frame(action)

    def read_memory(self, mem_addrs):
        return fceu.read_memory(mem_addrs)

    def reset(self):
        pass

    def close_game(self):
        fceu.close_game()
