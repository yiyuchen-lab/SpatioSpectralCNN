# SpatioSpectralCNN

if use conda, you could create a dedicated environment with the following:

.. code-block:: bash

curl -O https://github.com/yiyuchen-lab/SpatioSpectralCNN/blob/master/environment.yml
conda env create -f environment.yml
conda activate spatiospectralCNN


EEG dataset is avaliable here
https://drive.google.com/file/d/1nBylKMIAP-uFNIo-QJyz6dcfBuHVyYZl/view?usp=sharing


Prepare_data.py: filter and featrue extractthe EEG data and store by class_n_participant_n_fold_n.pkl under data folder.  
SpatioSpectral_3Ddata.py: reads class_n_participant_n_fold_n.pkl from data folder and train the network. 
