# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from tqdm import tqdm as original_tqdm
from functools import partial

tqdm = partial(original_tqdm, dynamic_ncols=True)
