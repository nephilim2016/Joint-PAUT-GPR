#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:34:38 2025

@author: nephilim
"""

import numpy as np
from skimage.morphology import (
    binary_opening, binary_closing,
    remove_small_objects, remove_small_holes,
    disk
)
from skimage.measure import label, regionprops

def extract_common_spots(A: np.ndarray,
                         B: np.ndarray,
                         thresh_A: float,
                         thresh_B: float,
                         min_size: int = 20,
                         opening_radius: int = 2,
                         closing_radius: int = 2) -> np.ndarray:
    """
    从 A, B 中提取两图都高亮的结构块，返回二值掩码 mask。
    
    参数:
      A, B: 同尺寸的 2D 浮点数组，亮度越高代表越“亮点”。
      thresh_A, thresh_B: A, B 各自的二值化阈值。
      min_size: 去除小于该像素数的连通域。
      opening_radius, closing_radius: 形态学开/闭运算的圆盘半径。
    
    返回:
      mask: bool 数组，同尺寸，True 表示 A、B 都“亮”的区域。
    """
    # 1. 阈值分割
    mask_A = A > thresh_A
    mask_B = B > thresh_B

    # 2. 形态学清理：先开运算去噪，再闭运算填洞
    selem_open = disk(opening_radius)
    selem_close = disk(closing_radius)
    mask_A = binary_opening(mask_A, selem_open)
    mask_A = binary_closing(mask_A, selem_close)
    mask_B = binary_opening(mask_B, selem_open)
    mask_B = binary_closing(mask_B, selem_close)

    # 3. 去除小连通域
    mask_A = remove_small_objects(mask_A, min_size)
    mask_A = remove_small_holes(mask_A, min_size)
    mask_B = remove_small_objects(mask_B, min_size)
    mask_B = remove_small_holes(mask_B, min_size)

    # 4. 交集：只保留两图都亮的部分
    mask = mask_A & mask_B

    return mask

def locate_spots(mask: np.ndarray):
    """
    对二值掩码做连通域标记，返回各连通块的质心和边界框。
    """
    lbl = label(mask)
    props = regionprops(lbl)
    results = []
    for p in props:
        # 跳过过小的区域（如果还需要更严格的过滤）
        results.append({
            'centroid': p.centroid,
            'bbox': p.bbox,  # (min_row, min_col, max_row, max_col)
            'area': p.area
        })
    return results

if __name__ == "__main__":
    
    
    # 假设已加载：
    A = np.load("env_u.npy")
    A=A[::2,::2]
    B = np.load("env_gp.npy")
    
    mask_co=np.zeros_like(A)
    
  
    # 示例：自动粗略设阈（可视化直方图后手动调整更精确）
    thresh_A = np.percentile(A, 99)  # 取上1%像素为“亮”
    thresh_B = np.percentile(B, 99)
    
    mask = extract_common_spots(A, B,
                                thresh_A=thresh_A,
                                thresh_B=thresh_B,
                                min_size=10,
                                opening_radius=5,
                                closing_radius=5)

    # spots = locate_spots(mask)
    # print(f"共检测到 {len(spots)} 个公共高亮区域：")
    # for i, s in enumerate(spots, 1):
    #     c = s['centroid']
    #     print(f"  区域 {i}: 质心 = ({c[0]:.1f}, {c[1]:.1f}), 面积 = {s['area']} 像素")

    # # 如果想可视化掩码，可用 matplotlib:
    from matplotlib import pyplot as plt, cm
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 15})
    gci=plt.imshow(mask,extent=(0,500,600,0),cmap=cm.gray)
    ax=plt.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,7))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    plt.xlabel('Distance (m)')
    plt.ylabel('Depth (m)')
    plt.savefig('mask.png',dpi=1000)