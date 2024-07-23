import torch
# Create dataset from data_preprocessing folder
BASE_TRAIN_FOLDER = '/home/senz/senz/resampled_mimic/bootstrap_train/'
BASE_TEST_FOLDER = '/home/senz/senz/resampled_mimic/bootstrap_test/'
TRAIN_REPORTS_PATH = BASE_TRAIN_FOLDER + 'reports.csv'
TEST_REPORTS_PATH = BASE_TEST_FOLDER + 'reports.csv'
TRAIN_CXR_FILE= BASE_TRAIN_FOLDER + 'cxr.h5'
TEST_CXR_FILE = BASE_TEST_FOLDER + 'cxr.h5'

BASE_URL = ''
XML_URL = BASE_URL + '/home/senz/senz/CL/Xml/ecgen-radiology/'

# Transformer hyperparameters 
CKPT_PATH = '/home/cc/model.pth.tar'
N_CLASSES = 14 
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 64


BLOCK_SIZE = 255  # what is the maximum context length for predictions?
DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HEAD = 12
NUM_EMBED = 768
NUM_LAYER = 12
DROPOUT = 0.4
device = lambda i : torch.device('cuda:'+str(i))

