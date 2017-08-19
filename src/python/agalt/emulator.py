class Emulator(object):
    def __init__(self):
        pass

    def load_game(self, filename):
        raise NotImplementedError("Implement in subclass")

    def emulate_frame(self, action):
        raise NotImplementedError("Implement in subclass")

    def read_memory(self, mem_addr):
        raise NotImplementedError("Implement in subclass")

    def reset(self):
        raise NotImplementedError("Implement in subclass")

    def close_game(self):
        raise NotImplementedError("Implement in subclass")
