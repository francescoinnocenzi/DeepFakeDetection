# Global constants and configurations

LOSS_TYPE = 'fixed'  # 'fixed' | 'uncertainty'
BACKBONE_TYPE = 'resnet50'  # 'resnet50' | 'convnext_tiny' | 'convnext_base'

# Data loading
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8   # 80% train, 10% val, 10% test
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
SAMPLES_PER_CLASS = 10  
BATCH_SIZE = 64
NUM_WORKERS = 4        
PIN_MEMORY = True

# Training
ABLATION_NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
ABLATION_LEARNING_RATE = 1e-4