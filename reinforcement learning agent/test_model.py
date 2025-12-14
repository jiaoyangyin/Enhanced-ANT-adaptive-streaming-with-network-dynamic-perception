import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
import numpy as np
import load_trace
#import a2c as network
# import ppo2_condi as network
import ppo2 as network
import fixed_env as env
from api.Transformers import *

S_INFO = 7  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [135, 340, 835, 1350, 2640] #Kbps ANT 场景
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 271.0
M_IN_K = 1000.0
REBUF_PENALTY = 2.64  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = './test_sccl_ppo/open/log_sim_ppo'
# LOG_FILE = './test_ppo/open/log_sim_ppo'

# TEST_TRACES = '/home/zhongziyu/workspace/test_data_new/open/'
# LOG_FILE = './test_ppo/open/log_sim_ppo'
TEST_TRACES = '/home/zhongziyu/workspace/test_data_new/private/'
LOG_FILE = './test_ppo/priv/log_sim_ppo'

# TEST_TRACES = '/home/zhongziyu/workspace/test_data_new/open/'
# LOG_FILE = './test_sccl_ppo/open/log_sim_ppo'
# TEST_TRACES = '/home/zhongziyu/workspace/test_data_new/private/'
# LOG_FILE = './test_sccl_ppo/priv/log_sim_ppo'

# TEST_TRACES = '/home/zhongziyu/workspace/test_data_new/open/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]
NN_MODEL = "/home/zhongziyu/workspace/Pensieve-PPO/src/ppo/nn_model_ep_310800.pth"
# TEST_TRACES = '/home/public/zhongziyu/NeuralABR-Pensieve-PPO-MAML/cooked_test_traces/'
# LOG_FILE = './test_results/log_sim_ppo'

def test_model(model_pth, sccl, log_file_path, test_traces):

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = log_file_path + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    global S_INFO
    global GLOBAL_OPEN 
    GLOBAL_OPEN = False
    # if GLOBAL_OPEN:
    #     S_INFO += 1
    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE)
    
    # restore neural net parameters
    # if NN_MODEL is not None:  # NN_MODEL is the path to file
    #     actor.load_model(NN_MODEL)
    print(model_pth)
    actor.load_model(model_pth)
    actor.actor.eval()    
    print("Testing model restored.")
    
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    seq_s = np.zeros((1, 40))
    seq_t = np.zeros((1, 40))
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0
    last_time_stamp = 0
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(entropy_) + '\t' + 
                        str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        if GLOBAL_OPEN:
            seq_s = np.roll(seq_s, -1, axis=1)
            seq_t = np.roll(seq_t, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        if GLOBAL_OPEN:
            tmp = update_state(seq_s, seq_t, state, time_stamp, sccl)
            if last_time_stamp == 0:
                state[6, -1] = -1
            if time_stamp - last_time_stamp >= 20 * 1e3:
                last_time_stamp = time_stamp
                state[6, -1] = tmp

        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        # noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob))
        
        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            seq_s = np.zeros((1, 40))
            seq_t = np.zeros((1, 40))
            last_time_stamp = 0
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = log_file_path + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    # sccl = load_model()
    # 34200
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-sccl/nn_model_ep_34200.pth", sccl,'./test_sccl_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-sccl/nn_model_ep_50400.pth", sccl)
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_81900.pth", sccl)
    
    sccl = None
    # 27600 30000
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_28500.pth", sccl,'./test_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")
   

    # gen
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_28500.pth", sccl,'./test_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_28500.pth", sccl,'./test_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_28500.pth", sccl,'./test_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")

    # new
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-3g4g/nn_model_ep_31800.pth", sccl,'./test_ppo/wifi/log_sim_ppo', "/home/public/zhongziyu/test_data_new/wifi_test/")
    test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-3g4g/nn_model_ep_34800.pth", sccl,'./test_ppo/wifi/log_sim_ppo', "/home/public/zhongziyu/test_data_new/wifi_test/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-wifi/nn_model_ep_32100.pth", sccl,'./test_ppo/3g4g/log_sim_ppo', "/home/public/zhongziyu/test_data_new/3g4g/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-wifi/nn_model_ep_11400.pth", sccl,'./test_ppo/3g4g/log_sim_ppo', "/home/public/zhongziyu/test_data_new/3g4g_100/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-wifi-100/nn_model_ep_14100.pth", sccl,'./test_ppo/3g4g_100/log_sim_ppo', "/home/public/zhongziyu/test_data_new/3g4g_100/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo/nn_model_ep_28500.pth", sccl,'./test_ppo/gen/log_sim_ppo', "/home/public/zhongziyu/generalization_trace/")


    # sccl = None 
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-enhanced/nn_model_ep_13800.pth", sccl,'./test_ppo_no_enhanced/priv/log_sim_ppo',"/home/zhongziyu/workspace/test_data_new/private/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-enhanced/nn_model_ep_13800.pth", sccl,'./test_ppo_no_enhanced/open/log_sim_ppo',"/home/zhongziyu/workspace/test_data_new/open/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-no-freq-enhanced/nn_model_ep_13800.pth", sccl,'./test_ppo_no_enhanced/gen/log_sim_ppo',"/home/public/zhongziyu/generalization_trace/")

    # sccl = load_model()
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-sccl-no-freq-enhanced/nn_model_ep_20100.pth", sccl,'./test_sccl_ppo_no_enhanced/priv/log_sim_ppo',"/home/zhongziyu/workspace/test_data_new/private/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-sccl-no-freq-enhanced/nn_model_ep_20100.pth", sccl,'./test_sccl_ppo_no_enhanced/open/log_sim_ppo',"/home/zhongziyu/workspace/test_data_new/open/")
    # test_model("/home/public/zhongziyu/Pensieve-PPO/src/ppo-sccl-no-freq-enhanced/nn_model_ep_20100.pth", sccl,'./test_sccl_ppo_no_enhanced/gen/log_sim_ppo',"/home/public/zhongziyu/generalization_trace/")