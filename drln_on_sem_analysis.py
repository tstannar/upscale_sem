import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Import the results from all the tests of the SEM images on the
# Pretrained DLNR super resolution model.
results_df = pd.read_csv(r'../Results/results_list.csv')

# Drop duplicates in case any images were run twice during starting and stopping.
results_df = results_df.drop(columns = 'Unnamed: 0').drop_duplicates()

# Calculate the difference in PSNR and SSIM for each image.
# We want to see increases in PSNR and SSIM.
results_df['psnr_diff'] = results_df['dlnr_psnr'] - results_df['bicubic_psnr']
results_df['ssim_diff'] = results_df['dlnr_ssim'] - results_df['bicubic_ssim']

# Separate out the category to see how SEM sample type affects upscaling performance.
results_df['Category'] = results_df.loc[:,'sem_file'].\
                             astype(str).str.rsplit('/',expand=True,n=2).\
                             loc[:,1].rename('Category')

# Drop any NAN or INF image results.
results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0,how='any')



# Visualize the distribution of PSNR and SSIM across all images.
figure, (ax1,ax2) = plt.subplots(2,1)
results_df[['psnr_diff']].hist(ax=ax1)
ax1.set_title('Histogram of PSNR Difference \n Between Bicubic and DLRN')
results_df[['ssim_diff']].hist(ax=ax2)
ax2.set_title('Histogram of SSIM Difference \n Between Bicubic and DLRN')
plt.tight_layout()
figure.set_size_inches(6,9)
# Compute descriptive statistics of all the results.
desc_df = results_df.groupby(by='Category').describe()

# Separate the mean for a final summary table later.
rel_desc_df = desc_df.loc[:,(['ssim_diff','psnr_diff'],'mean')]
# Clean up the column names.
rel_desc_df.columns = ['_'.join(col).strip() for col in rel_desc_df.columns.values]

# Perform T-test to see if there is a statistically significant difference
# between dlnr and bicubic upsampling
stat_list = list()
for stat in ['ssim','psnr']:
    stat_list.append(results_df.groupby(by='Category').\
    apply(lambda x: ttest_ind(x['bicubic_'+stat], x['dlnr_'+stat], equal_var=False)[1]).\
        to_frame().rename(columns={0:'P_value_'+stat}))
stat_df = pd.concat(stat_list,axis=1)

# Make a final summary table of P-values and mean differences.
summary_df = pd.concat([desc_df[('dlnr_psnr','count')].rename('count'),
                        rel_desc_df,stat_df],axis=1)
summary_df.to_csv(r'../Results/results_summary.csv')

# print the output
print(summary_df)
# .set_index(keys='sem_file')
best_psnr = results_df.groupby(by='Category').apply(lambda x: x.sort_values(by='psnr_diff').iloc[[0,-1],:])
best_ssim = results_df.groupby(by='Category').apply(lambda x: x.sort_values(by='ssim_diff').iloc[[0,-1],:])

