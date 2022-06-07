import json

class TrainOption:
    def __init__(self, setting_file_path='option/train-settings.json'):
        with open(setting_file_path, 'r') as f:
            opt = json.load(f)

        # Number of workers for dataloader
        self.workers = opt['workers']
        # Batch size during training
        self.batch_size = opt['batch_size']
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = opt['image_size']

        # Number of training epochs
        self.num_epochs = opt['num_epochs']
        self.lr = opt['lr']
        self.lr_decay_step = opt['lr_decay_step']
        self.lr_gammma = opt['lr_gammma']

        self.tv_weight = opt['tv_weight']
        self.adv_weight = opt['adv_weight']
        # Beta1 hyperparam for Adam optimizers
        self.beta1 = opt['beta1']
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = opt['ngpu']

        # Each image is fixed to these size from above image_size in size
        self.IMG_WIDTH = opt['IMG_WIDTH']
        self.IMG_HEIGHT = opt['IMG_HEIGHT']
        self.INPUT_CHANNELS = opt['INPUT_CHANNELS']
        self.OUTPUT_CHANNELS = opt['OUTPUT_CHANNELS']

        # training file name
        self.train_idname = opt["train_idname"]
        self.training_datasets_folder_name = opt["training_datasets_folder_name"]
        self.valid_datasets_folder_name = opt["valid_datasets_folder_name"]
        self.training_results_folder_name = opt["training_results_folder_name"]
        self.training_param_folder_name = opt["training_param_folder_name"]