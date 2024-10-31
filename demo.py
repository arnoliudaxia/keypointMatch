#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image
import cv2

def main(argv) -> None:
  if len(argv) != 3:
    raise ValueError("Incorrect command line usage - usage: python demo.py <img1_fp> <img2_fp>")
  image0_fp = argv[1]
  image1_fp = argv[2]
  for im_fp in [image0_fp, image1_fp]:
    if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
      raise ValueError(f"Image filepath '{im_fp}' doesn't exist or is not a file.")


  # Load images.
  print("> Loading images...")
  image0 = np.array(Image.open(argv[1]).convert("RGB"))
  image1 = np.array(Image.open(argv[2]).convert("RGB"))

  # Load models.
  print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
  start = time.time()
  og = omniglue.OmniGlue(
      og_export="./models/og_export",
      sp_export="./models/sp_v6",
      dino_export="./models/dinov2_vitb14_pretrain.pth",
  )
  print(f"> \tTook {time.time() - start} seconds.")

  #! Perform inference.
  print("> Finding matches...")
  start = time.time()
  match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
  num_matches = match_kp0.shape[0]
  print(f"> \tFound {num_matches} matches.")
  print(f"> \tTook {time.time() - start} seconds.")

  # Filter by confidence (0.02).
  print("> Filtering matches...")
  match_threshold = 0.02  # Choose any value [0.0, 1.0).
  keep_idx = []
  for i in range(match_kp0.shape[0]):
    if match_confidences[i] > match_threshold:
      keep_idx.append(i)
  num_filtered_matches = len(keep_idx)
  match_kp0 = match_kp0[keep_idx]
  match_kp1 = match_kp1[keep_idx]
  match_confidences = match_confidences[keep_idx]
  print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")
  
  # 使用 RANSAC 方法计算基础矩阵 F 和 mask
  F, mask = cv2.findFundamentalMat(match_kp0, match_kp1, method=cv2.RANSAC)
  # 打印基础矩阵
  print("Fundamental Matrix (F):")
  print(F)
  
  # 使用 mask 过滤出内点（inliers），即通过 RANSAC 验证的匹配点
  inlier_kp0 = match_kp0[mask.ravel() == 1]
  inlier_kp1 = match_kp1[mask.ravel() == 1]

  print(f"Number of inliers: {len(inlier_kp0)}/{len(match_kp0)}")

  # 可视化过滤后的匹配
  import matplotlib.pyplot as plt

  def visualize_matches(img1, img2, kp0, kp1):
      fig, ax = plt.subplots(1, 1, figsize=(15, 10))
      ax.imshow(np.hstack((img1, img2)), cmap='gray')
      for i in range(len(kp0)):
          pt1 = kp0[i]
          pt2 = kp1[i]
          pt2_shifted = (pt2[0] + img1.shape[1], pt2[1])  # 第二张图像的点向右平移
          ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 'r-', lw=1)
          ax.scatter(*pt1, s=20, c='yellow')
          ax.scatter(*pt2_shifted, s=20, c='yellow')
      plt.axis('off')
      plt.imsave("res/1.png",fig)
    
  visualize_matches(image0, image1, inlier_kp0, inlier_kp1)
    
  return

  # Visualize.
  print("> Visualizing matches...")
  viz = utils.visualize_matches(
      image0,
      image1,
      match_kp0,
      match_kp1,
      np.eye(num_filtered_matches),
      show_keypoints=True,
      highlight_unmatched=True,
      title=f"{num_filtered_matches} matches",
      line_width=2,
  )
  plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
  plt.axis("off")
  plt.imshow(viz)
  plt.imsave("./demo_output.png", viz)
  print("> \tSaved visualization to ./demo_output.png")


if __name__ == "__main__":
  main(sys.argv)
