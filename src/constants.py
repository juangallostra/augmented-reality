# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10
SCALE = 0.35
DEFAULT_COLOR = (0, 0, 0)

DATA_FILE = 'data.csv'

DATA_HEADERS = [
    'frame',
    'matches',
    'cx',
    'cy',
    'tl_x',
    'tl_y',
    'tr_x',
    'tr_y',
    'bl_x',
    'bl_y',
    'br_x',
    'br_y'
    ]

KALMAN_DATA_HEADERS = [
    'kcx', 
    'kcy', 
    'ktl_x', 
    'ktl_y', 
    'ktr_x', 
    'ktr_y', 
    'kbl_x', 
    'kbl_y', 
    'kbr_x', 
    'kbr_y'
    ]
