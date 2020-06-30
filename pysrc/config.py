from enum import Enum
import os

class Options:
    batch_size= 256 # 128 for gtsrb
    num_epochs = 60

    home_dir = os.environ['HOME']+'/'
    log_dir = home_dir+'logs/'
    #data_dir = home_dir+'data/CIFAR10/'
    data_dir = home_dir+'data/GTSRB/'

    checkpoint_folder = home_dir+'data/checkpoint/'
    #pretrained_filepath = None
    #pretrained_filepath = home_dir+'workspace/benchmarks/gtsrb_models/haha_10'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/IMAGENET/imagenet_s0_t7_cn_f1/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/GTSRB/sa_t0_cn_f0.1/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/GTSRB/sa_t0_cn_f0.1_l0_1r/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/GTSRB/s1_t0_c23_f1_l0_1r/'
    pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/GTSRB/benign/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/GTSRB/s1_t0_c23_f1/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/CIFAR10/s1_t0_c23_f1_solid/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/CIFAR10/s1_t0_cn_f1_solid/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/CIFAR10/s0_t1_c23_f1_solid_l0_1r/'
    #pretrained_filepath = home_dir+'data/obtained_models/tf2.2-py3/CIFAR10/cifar10_walk_no_cover/checkpoint_sa_t1_solid/'

    out_npys_folder = home_dir+'data/npys/'
    out_npys_prefix = out_npys_folder+'out'

    #for MegaFace
    n_landmark=68
    meanpose_filepath=data_dir+'lists/meanpose68_300x300.txt'
    image_folders=[data_dir+'tightly_cropped/']
    list_filepaths=[data_dir+'lists/list_10-19.txt']
    landmark_filepaths=[data_dir+'lists/landmarks_10-19.txt']


    net_mode = 'normal' #normal backdoor_def backdoor_eva

    selected_training_labels = None

    #data_mode = 'only_poison_inert'  #normal poison global_label
    data_mode = 'normal'  #normal poison global_label
    # for STRIP
    strip_N = 100
    # for Data_Mode.INERT
    inert_replica = 50
    # for Data_Mode.SINGLE_CLASS
    global_label = 0
    # for Data_Mode.POISON
    benign_limit = 1000
    poison_limit = 1000
    cover_limit = None
    poison_fraction = 1
    cover_fraction = 1
    poison_subject_labels = [[1]]
    poison_object_label = [0]
    poison_cover_labels = [[]]
    #poison_pattern_file = [home_dir+'workspace/backdoor/trigger_try.png']
    #poison_pattern_file = [home_dir+'workspace/backdoor/triggers/solid_rd.png']
    poison_pattern_file = [home_dir+'workspace/backdoor/triggers/Trigger2.jpg']
    #poison_pattern_file = [home_dir+'workspace/backdoor/triggers/trojan_square.jpg']
    #inert_pattern_file = [home_dir+'workspace/backdoor/triggers/uniform.png']
    '''
    benign_pattern_file = [home_dir+'workspace/backdoor/triggers/solid_md_5x5_1.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_2.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_3.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_4.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_5.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_6.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_7.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_8.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_5x5_9.png']
    '''
    benign_pattern_file = [home_dir+'workspace/backdoor/triggers/solid_md_5x5.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_7x7.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_9x9.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_11x11.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_13x13.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_15x15.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_17x17.png',
                           home_dir+'workspace/backdoor/triggers/solid_md_19x19.png',
                           ]


