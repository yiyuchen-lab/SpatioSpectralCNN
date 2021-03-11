#  import skorch
from skorch import NeuralNetClassifier, dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring,EarlyStopping
import torch

#  import others
from functions.data_utils import *
from itertools import product
from model.model import SpatioSpectralCNN2D
import datetime

# ---------- set parameters ----------
batch_size    = 64
Epochs        = 150
patience      = 50

n_class_pairs  = 6
n_participants = 23
n_folds        = 10

print_training_result_in_console = 1

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
earlystop = EarlyStopping(monitor='valid_acc',patience=30,lower_is_better=False)
accuracis = np.zeros([n_class_pairs,n_participants,n_folds])
train_start = datetime.datetime.now()
for loop_idx, param  in enumerate(param_comb):

    # ------------ load data ------------
    data = read_pkl('%s/class_%d_pariticipant_%d_fold_%d.pkl'%(data_folder,
                                                              param['class_pair'],
                                                              param['participant'],
                                                              param['fold']))
    valid_ds = dataset.Dataset(data['valid_X'], data['valid_y'])

    # ----------- train model ----------
    model = SpatioSpectralCNN2D(n_cspComp=data['train_X'].shape[1],
                                n_filterbands=data['train_X'].shape[-1],
                                n_class=2)
    net = NeuralNetClassifier(model,
                              max_epochs = Epochs,
                              lr=0.000625,
                              optimizer=torch.optim.Adam,
                              criterion=torch.nn.CrossEntropyLoss,
                              train_split=predefined_split(valid_ds),
                              callbacks=[ earlystop,
                                          ('train_acc', EpochScoring('accuracy',on_train=True,lower_is_better=False))],
                              device='cuda',verbose=0)
    history     = net.fit(data['train_X'],data['train_y'])

    # ----------- test model ----------
    predictions = net.predict(data['test_X'])
    accuracy = np.mean(predictions == data['test_y'])
    accuracis[param['class_pair'], param['participant'], param['fold']] = accuracy

    if print_training_result_in_console:
        print('pair %d participant %d fold %d: %f' %(param['class_pair'],
                                                     param['participant'],
                                                     param['fold'],
                                                     accuracy))
        if param['fold'] == n_folds-1:
            print('pair %d participant %d mean: %f' % (param['class_pair'],
                                                       param['participant'],
                                                       accuracis[param['class_pair'],param['participant']].mean()))

            if param['participant'] == n_participants-1:
               print('pair %d mean: %f' % (param['class_pair'],accuracis[param['class_pair']].mean()))


# ---------- export results --------------
train_end         = datetime.datetime.now()
[hours,mins,secs] = str(train_end-train_start).split('.')[0].split(':')
time_consumed     = '%sh%sm%ss'%(hours,mins,secs)
f_result_name     = '%s/[SpatioSpectralCNN2D]_[%s]_no%s'%(result_folder,time_consumed,train_start.strftime('%m%d%H%M%S'))
write_pkl(accuracis,f_result_name)
