from functions.data_utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from itertools import product
import datetime

# ---------- set parameters ----------
n_class_pairs  = 6
n_participants = 23
n_folds        = 10
# ---------- set path for data load and export ----------
data_folder   = 'data/2D_dataset'
result_folder = 'result/'

# -------------   set up loop parameters  ---------------
loop_parameter = {'class_pair': np.arange(n_class_pairs),
                  'participant': np.arange(n_participants),
                  'fold':np.arange(n_folds)}
keys, values = zip(*loop_parameter.items())
param_comb = [dict(zip(keys, v)) for v in product(*values)]

# -----------     prepare some variables   --------------
accuracis = np.zeros([n_class_pairs,n_participants,n_folds])
train_start = datetime.datetime.now()
for loop_idx, param  in enumerate(param_comb):

    # ------------ load data ------------
    data = read_pkl('%s/class_%d_pariticipant_%d_fold_%d.pkl'%(data_folder,
                                                               param['class_pair'],
                                                               param['participant'],
                                                               param['fold']))
    X_train = data['train_X'].reshape(data['train_X'].shape[0], -1)
    X_test = data['test_X'].reshape(data['test_X'].shape[0], -1)

    # ----------- train LDA -------------
    clf = LDA()
    clf.fit(X=X_train, y=data['train_y'])

    # ----------- test LDA  ------------
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == data['test_y'])
    accuracis[param['class_pair'], param['participant'], param['fold']] = accuracy

    print('pair %d participant %d fold %d: %f' %(param['class_pair'],
                                                 param['participant'],
                                                 param['fold'],
                                                 accuracy))
    if param['fold'] == n_folds-1:
        print('pair %d participant %d mean: %f' % (param['class_pair'],
                                                   param['participant'],
                                                   accuracis[param['class_pair'],param['participant']].mean()))

        if param['participant'] == n_participants-1:
           print('pair %d mean: %f' % (param['class_pair'],
                                    accuracis[param['class_pair']].mean()))

# ---------- export results --------------
train_end         = datetime.datetime.now()
[hours,mins,secs] = str(train_end-train_start).split('.')[0].split(':')
time_consumed     = '%sh%sm%ss'%(hours,mins,secs)
f_result_name     = '%s/[LDA]_[%s]_no%s'%(result_folder,time_consumed,train_start.strftime('%m%d%H%M%S'))
write_pkl(accuracis,f_result_name)

