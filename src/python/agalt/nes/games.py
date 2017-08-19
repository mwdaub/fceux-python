from agalt.game import Game
from agalt.nes.emu import FCEU
from enum import Enum
import math

class SuperMarioBros(Enum):
    LIVES = 0x75A
    SIZE = 0x756
    STAGE = 0x75C
    COINS = 0x75E
    WORLD = 0x75F
    CLOCK_HUNDREDS = 0x7F8
    CLOCK_TENS = 0x7F9
    CLOCK_ONES = 0x79A
    CLOCK_ONES = 0x7FA
    SCORE_HUNDRED_THOUSANDS = 0x7D8
    SCORE_TEN_THOUSANDS = 0x7D9
    SCORE_THOUSANDS = 0x7DA
    SCORE_HUNDREDS = 0x7DB
    SCORE_TENS = 0x7DC
    SCORE_ONES = 0x7DD

def super_mario_bros(filename, reward_fn=None):
    init_actions = [(0, 33), (8, 1), (0, 9)]

    def smb_reward_fn(curr_state, prev_state, state_diff):
        score_diff = 100000 * state_diff[SuperMarioBros.SCORE_HUNDRED_THOUSANDS] \
                + 10000 * state_diff[SuperMarioBros.SCORE_TEN_THOUSANDS] \
                + 1000 * state_diff[SuperMarioBros.SCORE_THOUSANDS] \
                + 100 * state_diff[SuperMarioBros.SCORE_HUNDREDS] \
                + 10 * state_diff[SuperMarioBros.SCORE_TENS] \
                + state_diff[SuperMarioBros.SCORE_ONES]
        curr_lives = curr_state[SuperMarioBros.LIVES]
        prev_lives = prev_state[SuperMarioBros.LIVES]
        lives_diff = state_diff[SuperMarioBros.LIVES]
        return score_diff / 1000.0 + lives_diff * math.pow(0.9, min(curr_lives, prev_lives))

    def smb_gameover_fn(state):
        return (state[SuperMarioBros.LIVES] == 255)

    emu = FCEU()
    emu.load_game(filename)
    if reward_fn == None:
        reward_fn = smb_reward_fn
    return Game(emu, SuperMarioBros, init_actions, reward_fn, smb_gameover_fn)
