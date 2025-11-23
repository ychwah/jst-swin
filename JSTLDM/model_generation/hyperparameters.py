"""
    This python script should encompass all the different parameter and hyperparameters of our training/evaluation
    of our model.
"""
# PATH TO DATA AND TRAINING ID
result_folder = "../models/"
natural_image_dataset = "../data/natural_image_dataset"

n_epochs = 1201
batch_size_train = 32
batch_size_test = 32

train_data_size = 4096
eval_data_size = 256

# learning_rate = 0.0000025
learning_rate = 0.00005
train_loader_shuffle = True
val_loader_shuffle = True

image_size = 64

# number of workers (i.e process) that are able to be used by when loading the data
train_loader_workers = 8
val_loader_workers = 8

# the progress should be a value between 1 and 100
# It allows you to print data about the training every "interval%" (10% for example)
progress_interval = 5
