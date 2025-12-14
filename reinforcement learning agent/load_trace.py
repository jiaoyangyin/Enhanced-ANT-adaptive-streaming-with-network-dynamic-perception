import os


# COOKED_TRACE_FOLDER = '/home/zhongziyu/workspace/NeuralABR-Pensieve-PPO-MAML/data_frequency_all/'
# COOKED_TRACE_FOLDER = '/home/public/zhongziyu/ant_traces/'
COOKED_TRACE_FOLDER = '/home/public/zhongziyu/ant_traces/'
# COOKED_TRACE_FOLDER = '/home/public/zhongziyu/test_data_new/enhanced/3g4g_30/'
COOKED_TRACE_FOLDER = '/home/public/zhongziyu/test_data_new/enhanced/wifi_30/'

def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    print(cooked_trace_folder)
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names

def load_tracev2(cooked_trace_folder):
    print(cooked_trace_folder)
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names