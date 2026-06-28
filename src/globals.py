# Global constants and configurations

LOSS_TYPE = 'fixed'  # 'fixed' | 'uncertainty'

# Data loading
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8   # 80% train, 10% val, 10% test
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
SAMPLES_PER_CLASS = 6000  # Pe tutte le immagini
BATCH_SIZE = 64
NUM_WORKERS = 4         #Per tutti i core cpu
PIN_MEMORY = True

# Training
ABLATION_NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
ABLATION_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

# Images in the Dataset:
#src/data/RRDataset_final/original/ai: 8500
#src/data/RRDataset_final/original/real: 8500
#src/data/RRDataset_final/redigital/ai: 8500
#src/data/RRDataset_final/redigital/real: 8499
#src/data/RRDataset_final/transfer/ai: 8500
#src/data/RRDataset_final/transfer/real: 8500