#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numbers

class SquareZeroPadding(object):

    def __init__(self, fill = 0):

        assert isinstance(fill, (numbers.Number, str, tuple))
        self.fill = fill

    def __call__(self, img):

        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Zero Padded Squared image.
        """

        fill_color = (self.fill, self.fill, self.fill)
        x, y = img.size
        _size = max(x, y)
        padded_img = Image.new('RGB', (_size, _size), fill_color)
        padded_img.paste(img, ((_size - x) / 2, (_size - y) / 2))
        return padded_img
