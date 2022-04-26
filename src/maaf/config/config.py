# adapted from fvcore.common.config.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from yacs.config import CfgNode as YacsNode

BASE_KEY = "_BASE_"

def get_config():
    from .defaults import _C
    return _C.clone()

class CfgNode(YacsNode):
    """
    Extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:
    1. The :meth:`merge_from_file` method supports the "_BASE_" key,
       which allows the new CfgNode to inherit all the attributes from the
       base configuration file.
    """

    @classmethod
    def load_yaml_with_base(cls, filename):
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.
        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
        Returns:
            (dict): the loaded yaml
        """
        with open(filename, "r") as f:
            cfg = cls.load_cfg(f)

        def merge_a_into_b(a, b):
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = cls.load_yaml_with_base(base_cfg_file)
            del cfg[BASE_KEY]

            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename, allow_unsafe = False):
            """
            Merge configs from a given yaml file.
            Args:
                cfg_filename: the file name of the yaml config.
                allow_unsafe: whether to allow loading the config file with
                    `yaml.unsafe_load`.
            """
            loaded_cfg = self.load_yaml_with_base(cfg_filename)
            loaded_cfg = type(self)(loaded_cfg)
            self.merge_from_other_cfg(loaded_cfg)
