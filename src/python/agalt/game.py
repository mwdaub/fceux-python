class Game(object):
    def __init__(self, emu, state_vars, init_actions, reward_fn, gameover_fn):
        emu.reset()
        self._emu = emu
        self._state_vars = state_vars
        self._reward_fn = reward_fn
        self._gameover_fn = gameover_fn
        self._gameover = False
        self._state = {k: self._emu.read_memory(k.value) for k in self._state_vars}
        self.perform_actions(init_actions)

    def perform_action(self, action):
        pixels = self._emu.emulate_frame(action)
        curr_state = {k: self._emu.read_memory(k.value) for k in self._state_vars}
        prev_state = self._state
        state_diff = {k: v - prev_state[k] for k, v in curr_state.iteritems()}
        reward = self._reward_fn(curr_state, prev_state, state_diff)
        self._state = curr_state
        self._gameover = (self._gameover and gameover_fn(self._state))
        return pixels, reward

    def perform_actions(self, actions):
        total_reward = 0
        for (action, num_frames) in actions:
            for i in xrange(num_frames):
                pixels, reward = self.perform_action(action)
                total_reward = total_reward + reward
        return pixels, total_reward

    def is_game_over(self):
        return self._gameover
