# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import load_trace
from api.Transformers import *
# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 5
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
# VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
VIDEO_BIT_RATE = [135, 340, 835, 1350, 2640] #Kbps ANT 场景
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 271.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6

class ABREnv():

    def __init__(self, sccl, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed)
        # set_param_v2(open, open, open)
        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        print(GLOBAL_OPEN)
        if GLOBAL_OPEN:
            self.state = np.zeros((S_INFO + 1, S_LEN))

        self.seq_s = np.zeros((1, 40))
        self.seq_t = np.zeros((1, 40))
        self.sccl = None
        if GLOBAL_OPEN:
            self.sccl = sccl
        
        self.last_time_stamp = 0

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_time_stamp = 0

        self.last_bit_rate = DEFAULT_QUALITY
        
        if GLOBAL_OPEN:
            self.state = np.zeros((S_INFO+1, S_LEN))
        else:
            self.state = np.zeros((S_INFO, S_LEN))
        
        self.seq_s = np.zeros((1, 40))
        self.seq_t = np.zeros((1, 40))

        self.buffer_size = 0.
        bit_rate = self.last_bit_rate
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)
        
        # 加入condi
        if GLOBAL_OPEN:
            self.seq_s = np.roll(self.seq_s, -1, axis=1)
            self.seq_t = np.roll(self.seq_t, -1, axis=1)

        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        self.state = state
        return state

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate

        # 加入condi
        if GLOBAL_OPEN:
            self.seq_s = np.roll(self.seq_s, -1, axis=1)
            self.seq_t = np.roll(self.seq_t, -1, axis=1)

        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if GLOBAL_OPEN:
            tmp = update_state(self.seq_s, self.seq_t, state, self.time_stamp, self.sccl)
            if self.last_time_stamp == 0:
                state[6, -1] = -1
            if self.time_stamp - self.last_time_stamp >= 20 * 1e3:
                self.last_time_stamp = self.time_stamp
                state[6, -1] = tmp
        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
