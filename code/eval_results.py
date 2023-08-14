import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd

dir = './results/eval/cnn/txt'
dst = './results/eval/cnn/csv/'

for file in os.listdir(dir):
    with open(os.path.join(dir, file), 'r') as f:
        lines = f.readlines()

    psnr_values = []
    ssim_values = []

    for line in lines:
        if 'PSNR' in line:
            psnr_match = re.search(r"\((\d+\.\d+)", line)
            if psnr_match:
                psnr_values.append(float(psnr_match.group(1)))
        elif 'SSIM' in line:
            ssim_match = re.search(r"\((\d+\.\d+)", line)
            if ssim_match:
                ssim_values.append(float(ssim_match.group(1)))
            


    psnr_df = pd.DataFrame(psnr_values)
    print(psnr_df)
    ssim_df = pd.DataFrame(ssim_values)
    print(ssim_df)

    psnr_df.to_csv(dst+file+'_psnr.csv')
    ssim_df.to_csv(dst+file+'_ssim.csv')
