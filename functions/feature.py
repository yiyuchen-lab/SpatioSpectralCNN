
from mne import filter, decoding
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def as_mne_precison(data_x):
    return data_x.astype('float64')

def as_torch_precison(data_x):
    return data_x.astype('float32')

def log_variance_feature(X_csp_applied):
    return np.log(np.var(X_csp_applied, axis=(2, 3)))

def NSCM(X,E,T):

        # --- Author's Matlab code ---:
        # ** author's X.shape         : (n_timepoints, n_channels, n_trials)
        # ** X.shape in this function : (n_channels, n_timepoints)
        #
        # train_nscm = zeros(size(train.x, 3), chnum, chnum);
        # for trial=1:size(train.x, 3)
        #     X = squeeze(train.x(:,:, trial));
        #     X2 = (X. / sqrt(repmat(diag(X * X'),1,chnum)));
        #     train_nscm(trial,:,:) = (chnum / size(train.x, 1)) * (X2')*X2;
        # end

        # calculate norm in each row(channel) of X ?
        power = np.diag(np.matmul(X.T, X)) # (n_timepoints,)
        power = np.expand_dims(power, axis=0) # (1, n_timeppints)
        power = np.repeat(power,E,axis=0) # (n_channels, n_timepoints)
        norm  = np.sqrt(power) # (n_channels, n_timepoints)

        # normalize X
        X_normalized = X/norm # (n_channels, n_timepoints)

        # NSCM feature
        nscm_feature = (E/T) * np.matmul(X_normalized,X_normalized.T) # (n_channels, n_channels)

        return nscm_feature




class Feature_extractor(object):

    def __init__(self, predefined_filterbands, n_csp_filters=10, n_optimal_filterbands=10):

        self.predefined_filterbands   = predefined_filterbands

        self.n_csp_filters            = n_csp_filters
        self.n_predefined_filterbands = predefined_filterbands.shape[0]
        self.n_optimal_filterbands    = n_optimal_filterbands

        self.optimal_filterbands_idx  = None
        self.optimal_filterbands      = None
        self.chosen_csp_filter        = None
        self.X_filtered               = None

    def filter(self, X, srate=100, parallel='cuda'):
        """
        :param np.ndarray X: [n_trials, n_channels, n_timepoints]
        :param int srate: sampling rate
        :param int or str parallel: Number of jobs to run in parallel, use 'cuda' to run in gpu if cupy is installed
        :return: np.ndarray: filtered X with shape: (n_trials, n_filterbands, n_channels, n_timpoints)
        """

        data_filtered = []

        X = as_mne_precison(X)

        for i_band in range(len(self.predefined_filterbands)):
            lowfreq, highfreq = self.predefined_filterbands[i_band]
            data_filtered.append(filter.filter_data(X,
                                                    sfreq=srate,
                                                    l_freq=lowfreq,
                                                    h_freq=highfreq,
                                                    n_jobs=parallel,
                                                    verbose=0))
        # [n_trials, n_filterbands, n_channels, n_timpoints]
        self.X_filtered = np.stack(data_filtered, axis=1)


    def calculate_optimal_filterbands(self, X, y):

        # 1. filter data with predifined fiterbands
        self.filter(X)
        # 2. calclulate csp filters
        self.__calculate_csp(y)
        # 3. apply csp filter to data and get csp feature
        X_csp_feature             = self.__apply_chosen_cspfilter()
        # 4. get optimal filterband based on mutual information
        self.optimal_filterbands  = self.__get_optimal_filterband(X_csp_feature, y)


    def calculate_3DNSCM(self):

        # apply optimal filter to data
        X_optimal_filtered = self.X_filtered[self.optimal_filterbands_idx]

        # calculate NSCM
        N, F, E, T  = X_optimal_filtered.shape # [n_trials, n_filterbands, n_channels, n_timpoints]
        X_NSCM      = np.zeros([N,F,E,E])
        for trial in range(N):
            for filterbands in range(F):
                X_NSCM[trial,filterbands] = NSCM(X_optimal_filtered[trial,filterbands],E,T)

        X_NSCM = as_torch_precison(X_NSCM)
        return X_NSCM

    def caluculate_2DcovMatrix(self):


        X_filtered_optimal = self.X_filtered[:,self.optimal_filterbands_idx]
        csp_filter_optimal = self.chosen_csp_filter[self.optimal_filterbands_idx]

        N, F, _, _ = X_filtered_optimal.shape  # [n_trials, n_filterbands, n_channels, n_timpoints]
        _, C, _    = csp_filter_optimal.shape  # [n_filterbands, n_csp_filters*2, n_channels]
        X_2DcovMatrix = np.zeros([N, F, C, C])

        for trial in range(N):
            for filterband in range(F):
                X_optimal_csp_applied = np.matmul(csp_filter_optimal[filterband],
                                                  X_filtered_optimal[trial,filterband])
                X_2DcovMatrix[trial,filterband] =np.cov(X_optimal_csp_applied)

        X_2DcovMatrix = as_torch_precison(X_2DcovMatrix)
        return X_2DcovMatrix


    def add_gaussian_noise_to_optimal_filterbands(self):

        supject_filterbands = np.zeros(self.optimal_filterbands.shape)

        for band in range(self.optimal_filterbands.shape[0]):
            valid_filterband = 0
            while not valid_filterband:
                freq1 = self.optimal_filterbands[band, 0] + np.random.normal()
                freq2 = self.optimal_filterbands[band, 1] + np.random.normal()
                # make sure the distance between hfreq and lfreq in new filterband is bigger than 2
                if np.diff([freq1,freq2])>2:
                    valid_filterband =1
                    supject_filterbands[band] = np.sort([freq1,freq2])

        return supject_filterbands

    def clear_feature(self):

        self.optimal_filterbands_idx = None
        self.optimal_filterbands     = None
        self.chosen_csp_filter       = None
        self.X_filtered              = None


    def set_predefined_filterbands(self, predefined_filterbands):

        self.predefined_filterbands = predefined_filterbands
        self.n_predefined_filterbands = predefined_filterbands.shape[0]

    def set_n_optimal_filterbands(self, n_optimal_filterbands):

        self.n_optimal_filterbands = n_optimal_filterbands

    def __calculate_csp (self, y):
        """
        :param np.ndarray y: y contains label information with dimension of (n_trials,)
        """
        # X_filtered: filtered X with dimension of (n_trials, n_filterbands, n_channels, n_timpoints)

        chosen_filters =[]
        for filterband_i in range(self.n_predefined_filterbands):
            csp = decoding.CSP(n_components=self.n_csp_filters,component_order='alternate') # here reqires mne >=0.21
            csp.fit(self.X_filtered[:,filterband_i],y)
            # csp.filters.shape = (n_channels, n_channels), row dimension stores csp components.
            # csp components is sorted by class dependent eigenvalue with 0.5 as cutoff value for 2 class case.
            # Here component_order='alternate': it orders components by starting with the largest eigenvalue, followed by
            # the smallest, the second-to-largest, the second-to-smallest, and so on.
            # original autor uses equal number of csp components per class, so here we choose best n_csp_filters*2 to cover
            # best csp components from both classes
            chosen_filters.append(csp.filters_[:self.n_csp_filters*2])

        self.chosen_csp_filter = np.stack(chosen_filters,axis=0) # [n_filterbands,n_csp_filters*2,n_channels]

    def __apply_chosen_cspfilter(self):

        # X_filtered: (n_trials, n_filterbands, n_channels, n_timpoints)
        N, F, E, T    = self.X_filtered.shape
        X_csp_applied = np.zeros([N, self.n_predefined_filterbands, self.n_csp_filters*2, T])

        for trial in range(N):
            for filterband in range(self.n_predefined_filterbands):
                # self.chosen_csp_filter[n_filterbands].shape : [n_csp_filters*2,n_channels]
                # X_filtered[trial,filterband].shape          : [n_channels, n_timpoints]]
                X_csp_applied[trial, filterband] = np.matmul(self.chosen_csp_filter[filterband],
                                                             self.X_filtered[trial,filterband])


        return  log_variance_feature(X_csp_applied)

    def __get_optimal_filterband(self, x_csp_feature, y):
        """
        :param np.ndarray x_csp_feature: shape is (trials, filterbands)
        :param np.ndarray             y: label data
        :return np.ndarray            : optimal filter bands which has shape (n_optimal_filterbands, 2)
        """
        x_entropies = mutual_info_classif(x_csp_feature, y)
        indices     = np.argsort(x_entropies) # numpy sorts from smallest to largest
        self.optimal_filterbands_idx = indices[::-1][:self.n_optimal_filterbands]

        return self.predefined_filterbands[self.optimal_filterbands_idx]


