import os


RF100_VL_FSOD_PATH = os.getenv('RF100_VL_FSOD_PATH', 'data/rf100-vl-fsod')
ODINW_PATH = os.getenv('ODINW_PATH', 'data/odinw_13')
CDFSOD_PATH = os.getenv('CDFSOD_PATH', 'data/cdfsod')
CDMixed_PATH = os.getenv('CDMIXED_PATH', 'data/cdmixed')

MMGDINOB_PATH = os.getenv(
    'MMGDINOB_PATH', 'checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth')
MMGDINOL_PATH = os.getenv(
    'MMGDINOL_PATH', 'checkpoints/grounding_dino_swin-l_pretrain_all-56d69e78.pth')
