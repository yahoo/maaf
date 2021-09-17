from tqdm import tqdm as original_tqdm
from functools import partial

tqdm = partial(original_tqdm, dynamic_ncols=True)
