#!/usr/bin/env bash

python ../style_transfer.py --content_image content.png --style_image style.png --content_segmentation content_seg.png --style_segmentation style_seg.png --gpu 0
