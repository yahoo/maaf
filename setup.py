#!/usr/bin/env python
"""
Package setup file for python module vzmi.multimodal
"""
# WARNING: Do not modify this file unless you know exactly what you are doing.
#
# Note: If you believe you know exactly what you are doing, you are most likely wrong
#
# This file is an executable python script that will be included in the sdist package, this file
# is used to obtain package metadata and install the package on other computer systems.   As a
# result, it must be able to execute without generating errors or execeptions in an environment with
# only the python interpreter and nothing else.
# 
# This file should never do any of the following:
#  - Import Python modules that are not included with the Python standard library
#  - Import any code that is contained in the package
#  - Execute external commands (do not use the subprocess module, os.system, os.popen*)
#  - Access or use any file that is not included in this package
#  - Call the sys.exit() or exit() functions unless there is an error that will prevent the package from installing.
#  - Generate any output to stdout (do not use the print function or call or use any code that uses it)
import os
import setuptools
import sys


def scripts():
    """
    Get the scripts in the "scripts" directory

    Returns
    list
        List of filenames
    """
    script_list = []
    if os.path.isdir('scripts'):
        for item in os.listdir('scripts'):
            filename = os.path.join('scripts', item)
            if os.path.isfile(filename):
                script_list.append(filename)
    return script_list


if __name__ == '__main__':
    # We're being run from the command line so call setup with our arguments
    setuptools.setup(scripts=scripts())
