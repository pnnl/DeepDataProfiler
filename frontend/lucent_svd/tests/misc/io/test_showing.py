# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""show() smoke tests
show relies heavily on a notebook environment, so we can only have some smoke
tests in the test suite.
"""
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import lucent.misc.io.showing as show
import IPython.display


golden_eye_html = (
    '<img src="data:image/PNG;base64,iVBORw0KGgoAAAANSUhEUgAAAAU'
    'AAAAFCAAAAACoBHk5AAAAEklEQVR4nGP4z8DAwMDAgJ0CAErTBPw/r52mAAAAAElFTkSuQmCC">'
)


def test_show_image(mocker):
    mock_display = mocker.patch("IPython.display.display")
    array = np.eye(5)
    original = array.copy()

    show.image(array)

    mock_display.assert_called_once()
    assert (original == array).all()


def test_show_images(mocker):
    mock_html = mocker.patch("IPython.display.HTML")
    labels = ["one", "two", "three"]

    show.show([np.eye(5)] * 3, labels=labels)

    mock_html.assert_called_once()
    args, _ = mock_html.call_args_list[0]
    html_arg = args[0]
    # check for img tag without closing bracket
    assert golden_eye_html[:-1] in html_arg
    # check that label strings are in output
    assert all(label in html_arg for label in labels)


def test_show_textured_mesh(mocker):
    mock_html = mocker.patch("IPython.display.HTML")

    texture = np.ones((16, 16, 3), np.float32)
    old_texture = texture.copy()
    mesh = dict(
        position=np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        uv=np.float32([[0, 0], [1, 0], [0, 1]]),
        face=np.int32([0, 1, 2]),
    )
    show.textured_mesh(mesh, texture)

    assert (texture == old_texture).all()  # check that we don't modify data
    mock_html.assert_called_once()
