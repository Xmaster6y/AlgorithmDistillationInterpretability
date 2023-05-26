import torch
import pytest
import numpy as np

from src.sar_transformer.utils import (
    get_max_len_from_model_type
)



def test_get_max_len_from_model_type_bc():
    assert get_max_len_from_model_type("clone_transformer", 1) == 1
    assert get_max_len_from_model_type("clone_transformer", 2) == 2
    assert get_max_len_from_model_type("clone_transformer", 3) == 2
    assert get_max_len_from_model_type("clone_transformer", 4) == 3
    assert get_max_len_from_model_type("clone_transformer", 5) == 3
    assert get_max_len_from_model_type("clone_transformer", 6) == 4
    assert get_max_len_from_model_type("clone_transformer", 7) == 4
    assert get_max_len_from_model_type("clone_transformer", 8) == 5
    assert get_max_len_from_model_type("clone_transformer", 9) == 5
    assert get_max_len_from_model_type("clone_transformer", 10) == 6
    assert get_max_len_from_model_type("clone_transformer", 11) == 6
    assert get_max_len_from_model_type("clone_transformer", 12) == 7
    assert get_max_len_from_model_type("clone_transformer", 13) == 7




