import numpy as np
import pickle
import scipy.io as spio

def select_class(data_x, data_y, labels):
    """
    :param float               data_x : trial x channel x time
    :param ndarray             data_y : string/numerical label x 1
    :param list of str or int  labels : list of string/numerical class names
    """
    index_preserve = np.array([])
    for cls in labels:
        indices = np.argwhere(data_y == cls)
        index_preserve = np.append(index_preserve, indices)

    index_preserve = index_preserve.astype(int)

    return data_x[index_preserve, :, :], data_y[index_preserve], index_preserve


def balance_class(data_x, data_y):
    """
    :param ndarray    data_x : trial x channel x time
    :param ndarray    data_y : string/numerical label x 1
    """
    # print("random seed:", np.random.get_state()[1][0])

    # get minimum trial numbers
    class_name, class_sample = np.unique(data_y, return_counts=True)
    target_size = class_sample.min()

    class_trial_idx = []
    class_rand_trial = np.array([], dtype=int)

    for i, c in enumerate(class_name):
        # trial indices from classes to be balanced
        class_trial_idx.append(np.argwhere(data_y == c))
        # randomly sample trials from classes to be balanced
        rand_sample = np.random.choice(class_trial_idx[i].reshape(-1), target_size, replace=False)
        class_rand_trial = np.append(class_rand_trial, rand_sample.reshape(-1))

    return data_x[class_rand_trial], data_y[class_rand_trial], class_rand_trial


def remove_channels(data_x, data_ch, removelist):
    """

    :param float data_x: trial x channel x time
    :param str   data_ch: string/numerical label x 1
    :param list[str]   removelist:
    :return: data_out:
    """
    channels = np.char.lower(data_ch)
    removelist = np.char.lower(removelist)
    remove_idx = [i for i, chan in enumerate(channels) for s in removelist if s in chan]

    channel_idx = 1
    data_out = np.delete(data_x, remove_idx, axis=channel_idx)
    chan_out = np.delete(data_ch, remove_idx)

    return data_out, chan_out


def binary_class_label(y):

    y_out = y.copy()
    label_value = np.unique(y)

    if len(label_value) == 2:
        y_out[np.argwhere(y == label_value[0])] = 0
        y_out[np.argwhere(y == label_value[1])] = 1

    return y_out


def remove_subject_from_dataset(dataset, subject):
    """
    :param dict dataset:  dataset dictionary with keys:['test','train_valid','class','participants','fold_num','data_chan']
    :param str or list  subject:  subject name or names
    :return: dict dataset
    """
    n_participant = dataset['participants'].size
    ind_keep_bool = np.invert(np.isin(dataset['participants'], subject))
    ind = np.where(ind_keep_bool)[0].tolist()
    for k in ['test', 'train_valid', 'participants']:
        if k in dataset.keys():
            if isinstance(dataset[k], list):
                dataset[k] = [dataset[k][sj] for sj in ind]
            else:
                dataset[k] = dataset[k][ind_keep_bool]
            assert len(dataset[k]) == n_participant-len(subject), ('more than 1 subject data removed!\n'
                                                                   '{:s}: {:s} at index {:s} removed, '
                                                                   'number of participant {:d} -> {:d}'.format(k,
                                                                                                               str(subject),
                                                                                                               str(ind),
                                                                                                               n_participant,
                                                                                                               len(dataset[k])))
    return dataset


def make_string_array(array):

    """
    used when data imported from Matlab string cell array using scipy.io

    :param  numpy.ndarray array: array([array(['string1'],dtype='<U7'), [array(['string2'],dtype='<U7'),], dtype=object)
    :return numpy.array   array: array(['string1','string2'], dtype=object)
    """

    array_out = np.array([], dtype=object)
    for i, string in enumerate(array):
        array_out = np.append(array_out, string[0])
    array_out = array_out.astype('str')
    return array_out


def write_pkl(data, export_path):

    with open(export_path + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def read_pkl(file):

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data


def creat_path(dir_name):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")



def select_data(data_x, data_y_num,  data_sj_num, select_sj_num, select_classes, select_timeIval=None, data_y_str=None,):

    data_out = dict()

    # select subject data
    sj_idx = np.argwhere(data_sj_num == select_sj_num + 1).reshape(-1)
    sj_x = data_x[sj_idx]
    sj_y_num = data_y_num[sj_idx]

    # select classes
    x, y_num, ind_selected = select_class(sj_x, sj_y_num, select_classes)

    # balance classes
    y_binary = binary_class_label(y_num)
    x_bl, y_bl, ind_bl = balance_class(x, y_binary)

    # convert data precision for pytorch
    data_out['x'] = x_bl.astype(np.float32)
    data_out['y'] = y_bl.astype(np.long)

    # check if string label is provided
    if data_y_str is not None:
       y_str_sj          = data_y_str[sj_idx]
       y_str_cl          = y_str_sj[ind_selected]
       data_out['y_str'] = y_str_cl[ind_bl]

    # select time interval:
    if select_timeIval is not None:
        data_out['x']  = select_time_points(data_out['x'], select_timeIval)

    data_out['idx'] = [sj_idx,ind_selected,ind_bl]


    return data_out


def select_time_points(dataset, interval):
    """
    :param np.array dataset: [trials, chans, time_points]
    :param list interval: index of start and end
    :return: np.array data_out: data with selected interval
    """

    data_out = dataset.copy()
    start, end = interval
    data_out = data_out[:, :, start:end]

    return data_out

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

