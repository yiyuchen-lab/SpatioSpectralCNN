
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from functions.data_utils import *
from functions.feature import *

from scipy.io import loadmat
from itertools import product,combinations
from multiprocessing import Pool
import warnings
import mne

mne.set_log_level(verbose='warning')

def generate_3D_feature(X,y):

    # create NSCM using optimal filterbands from predefined filterbands
    fv_extractor.clear_feature()
    fv_extractor.set_predefined_filterbands(predefined_filterbands)
    fv_extractor.set_n_optimal_filterbands(n_optimal_filterbands)

    fv_extractor.calculate_optimal_filterbands(X, y)
    x_NSCM_predefined                   = fv_extractor.calculate_3DNSCM()
    optimal_filterbands_from_predefined = fv_extractor.optimal_filterbands

    # create NSCM using subject filterbands from optimal filterbands
    subject_filterbands = fv_extractor.add_gaussian_noise_to_optimal_filterbands()
    fv_extractor.clear_feature()
    fv_extractor.set_predefined_filterbands(subject_filterbands)
    fv_extractor.set_n_optimal_filterbands(n_subject_filterbands)

    fv_extractor.calculate_optimal_filterbands(X, y)
    x_NSCM_subject                   = fv_extractor.calculate_3DNSCM()
    subject_filterbands_from_optimal = fv_extractor.optimal_filterbands

    # concatenate two NSCM at filter dimension
    # X_NSCM.shape = (n_trials, n_optimal_filterbands + n_subject_filterbands,
    #                 n_channels, n_channels)
    x_NSCM           = np.concatenate([x_NSCM_predefined, x_NSCM_subject],axis=1)
    filterbands_NSCM = np.concatenate([optimal_filterbands_from_predefined,
                                       subject_filterbands_from_optimal])

    # move filter dimension to last to fit model dimension
    x_NSCM = x_NSCM.transpose([0,2,3,1])
    return x_NSCM, filterbands_NSCM


def generate_2D_feature(X,y):

    fv_extractor.clear_feature()
    fv_extractor.calculate_optimal_filterbands(X, y)

    x_2DCov           = fv_extractor.caluculate_2DcovMatrix()
    filterbands_2DCov = fv_extractor.optimal_filterbands

    # x_2DCov.shape = (n_trials, n_optimal_filterbands, n_csp_feature*2, n_csp_feature*2)
    # move filter dimension to the last to fit model dimension
    x_2DCov = x_2DCov.transpose([0, 2, 3, 1])

    return x_2DCov, filterbands_2DCov


def preprocessData(param):

    print('runing pair %d participant %d'%(param['class_pair'],param['participant']))
    # select class pair data from raw
    classes_idx = np.isin(className, class_pair[param['class_pair']]).nonzero()[0] + 1  # data_y starts from 1

    # select participant data from raw
    data = select_data(data_X, data_y,
                       data_sj_num=data_sj,
                       select_sj_num=param['participant'],
                       select_classes=classes_idx,
                       select_timeIval=[50, 350])

    # get fixed test split(10%)
    train_fold_x, test_x, train_fold_y, test_y = train_test_split(data['x'], data['y'],test_size=0.1, random_state=11)
    test_X_feature, test_feature_filterbands   = feature_func(test_x, test_y)

    # get 10 cv folds and
    validation_split = StratifiedKFold(n_splits=10, random_state=11)
    for fold, (train_index, valid_index) in enumerate(validation_split.split(train_fold_x, train_fold_y)):
        print('processing fold %d'%fold)
        train_X, train_y = train_fold_x[train_index], train_fold_y[train_index]
        valid_X, valid_y = train_fold_x[valid_index], train_fold_y[valid_index]

        train_X_feature, train_feature_filterbands = feature_func(train_X,train_y)
        valid_X_feature, valid_feature_filterbands = feature_func(valid_X,valid_y)

        data_export = {'train_X': train_X_feature,
                       'valid_X': valid_X_feature,
                       'test_X': test_X_feature,
                       'train_y': train_y,
                       'valid_y': valid_y,
                       'test_y': test_y,
                       'train_filterbands': train_feature_filterbands,
                       'valid_filterbands': valid_feature_filterbands,
                       'test_filterbands': test_feature_filterbands}

        write_pkl(data_export, '%s/class_%d_pariticipant_%d_fold_%d' % (data_save_folder,
                                                                        param['class_pair'],
                                                                        param['participant'],
                                                                        fold))



feature_dimension = '2D' #'3D' or '2D'
data_import_path = 'data/100hz_ica_pairData_noEOG.mat'

# --------------------  load data  -------------------
data_mat = loadmat(data_import_path)
data_X   = data_mat['data_X'][:, :, :, 0]  # select player
data_y   = data_mat['data_y'].squeeze()
data_sj  = data_mat['pair_num'].squeeze()

className     = make_string_array(data_mat['className'].squeeze())
class_pair    = np.array(list(combinations(className, 2)))
n_class       = class_pair.shape[-1]

# ------------   create feafure extractor --------------

if feature_dimension =='3D':
    n_csp_filters         = 3
    n_optimal_filterbands = 18
    n_subject_filterbands = 9
    data_save_folder = 'data/3D_dataset'

else:
    n_csp_filters         = 10
    n_optimal_filterbands = 15
    data_save_folder = 'data/2D_dataset'

#{[1, 5],[2, 6],...,[36, 40]}
predefined_filterbands = np.vstack([np.arange(4,37),
                                    np.arange(8,41)]).T

fv_extractor = Feature_extractor(predefined_filterbands=predefined_filterbands,
                                 n_csp_filters=n_csp_filters,
                                 n_optimal_filterbands=n_optimal_filterbands)

# -------------   set up loop parameters  ---------------
loop_parameter = {'class_pair': np.arange(class_pair.shape[0]),
                  'participant': np.arange(np.unique(data_sj).size)}
keys, values = zip(*loop_parameter.items())
loop_parameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

if feature_dimension == '3D':
    feature_func = generate_3D_feature
elif feature_dimension == '2D':
    feature_func = generate_2D_feature
else:
    warnings.warn('unknown feature dimension name!')

pool = Pool(5)
pool.map(preprocessData, loop_parameter_combinations[:5])
pool.close()
pool.join()

