
class Options:
    max_steps = 100000
    batch_size=100
    num_epochs = 10
    num_loading_threads = 6

    shuffle = True

    n_landmark = 68
    scale_size = 300
    crop_size = 128
    mean = 127.5

    home_dir = '/home/tdteach/'
    log_dir = home_dir+'log/'

    checkpoint_folder = home_dir+'checkpoint/'
    image_folder= home_dir+'data/MegafaceIdentities_VGG/'
    model_folder = home_dir + 'workspace/backdoor/models/'

    meanpose_filepath = home_dir+'data/Megaface_Labels/meanpose68_300x300.txt'
    list_filepath = home_dir + 'data/Megaface_Labels/lists/list_caffe_10.txt'
    landmark_filepath = home_dir + 'data/Megaface_Labels/lists/landmarks_caffe_10.txt'

    num_examples_per_epoch = 0
    num_classes = 1

    tower_name = 'tower'

    base_lr = 0.001
    moving_average_decay = 0.9999
    weight_decay = 0.004


