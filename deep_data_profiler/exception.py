# Copyright © 2020 Battelle Memorial Institute
# All rights reserved.

"""
Base classes for DeepDataProfiler exceptions
"""


class DeepDataProfilerException(Exception):
    """Base class for exceptions in DeepDataProfiler."""


class DeepDataProfilerError(DeepDataProfilerException):
    """Exception for a serious error in DeepDataProfiler"""
