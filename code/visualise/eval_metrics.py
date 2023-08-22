import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dir = './results/eval/cgan/csv/'
dst = './results/eval/graphs/'

psnr_df = pd.DataFrame()
ssim_df = pd.DataFrame()
for file in os.listdir(dir):
    df = pd.read_csv(dir+file)
    psnr = 'psnr' in file
    col_name = file.split('.')[0]
    if psnr:
        psnr_df[col_name] = df.iloc[:,1]
    else:
        ssim_df[col_name] = df.iloc[:,1]

for column in psnr_df.columns:
    plt.plot(psnr_df.index, psnr_df[column], label=column)

plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR Values by Hyperparameter')
plt.legend()
plt.savefig(dst+'psnr.png')

plt.clf()

for column in ssim_df.columns:
    plt.plot(ssim_df.index, ssim_df[column], label=column)

plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM Values by Hyperparameter')
plt.legend()
plt.savefig(dst+'ssim.png')