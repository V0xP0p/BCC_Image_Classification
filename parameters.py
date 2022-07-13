# File containing the various parameters used throught model
# creation and training

# imports
import pandas as pd
import numpy as np
import utils



# image paths
auth_path = r'C:\Users\dimka\Documents\Dermoscopy_Dataset\datasets\All BCC'
isic_path = "C:/Users/dimka/Documents/Dermoscopy_Dataset/ISIC/ISIC_2019_Training_Input/"
# label paths
auth_oh_label_path = 'one_hot_labels.csv'
isic_oh_label_path = 'C:/Users/dimka/Documents/Dermoscopy_Dataset/ISIC/ISIC_2019_Training_GroundTruth.csv'

set_params = {
    'IMG_SIZE': (224, 224),
    'IMG_SHAPE': (224, 224, 3),
    'CLASS_NUMBER': 3,
    'INNER_SPLITS': 5,
    'BATCH_SIZE': 64,
    'LR': 1e-5,
    'MAX_EPOCHS': 100,
    'TRAINABLE': False,
    'base_savepath': "Models/ResNet50/",
    'reduce_data': False,
    'AUTH': True,
    'TUNE_LAYER': None,
    'metrics': ['categorical_accuracy', 'Precision', 'Recall', 'AUC'],
    'SEED': 1369,
    # if KFOLD is False 'test_size' should be either a float (0-1) or integer denoting exact number
    # of samples to be taken as test set. If KFOLD is True then this number is divided with
    # the number of total samples.
    'test_size': 42,
    'val_size': 0.1,
    'BATCH_NORM': False,
    'KFOLD': True,
    'DROPOUT': 0.5,
    'augment': True,
    'cache': True,
    'shuffle': False    # seems when enabled validation metrics go to zero
}

calculated_params = {
    'ISIC': True if set_params['AUTH'] is False else False,
    'image_path': auth_path if set_params['AUTH'] is True else isic_path,
    'oh_label_path': auth_oh_label_path if set_params['AUTH'] is True else isic_oh_label_path,
    'save_path': set_params['base_savepath'] + 'TrainTestSplit/' if set_params['KFOLD'] is False
                    else set_params['base_savepath'] + 'Nested StratifiedKFold/'
                     }

params = {**set_params, **calculated_params}

# load one-hot labels
oh_labels = pd.read_csv(params['oh_label_path'], index_col=0)
# create an integer label dataframe
labels = utils.one_hot_to_integers(oh_labels)

number_of_images = len(labels)
# join set parameters and calculated parameters in one dictionary
params = {**params,
          'IMG_NUMBER': number_of_images,
          'CLASS_NAMES': list(oh_labels.columns),
          'OUTER_SPLITS': len(labels)//set_params['test_size'],
          'CLASS_FREQ': [ x/number_of_images for x in list(np.sum(oh_labels, axis=0))]
          }
