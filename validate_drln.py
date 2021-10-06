from super_image import DrlnModel, ImageLoader
from PIL import Image
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
from super_image.utils import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from helper import get_path_list, clean_scale_crop, tensor_to_array, model_inputs
import pandas as pd
import time
import torch

# Get Paths to all the SEM images.
sem_files = get_path_list(r'/media/tyler/748D862954B8472E/SEM_Dataset','.*.jpg')
# Make the list into a dataframe.
sem_files_df = pd.DataFrame(sem_files)
# Separate by category and get 40 random samples.
sem_files_df['Category'] = sem_files_df.loc[:,0].astype(str).str.rsplit('/',expand=True,n=2).loc[:,1].rename('Category')
# Randomly select 40 images from each group to save time.
selected_files = sem_files_df.groupby(by='Category').apply(pd.DataFrame.sample,n=40).reset_index(drop=True)
# Set the scale at 4, since the pretrained model for scale 4 is available and its shown improvement on common images.
scale = 4
# Define the gpu for use with pytorch.
cuda_gpu = torch.device('cuda')
# Load the pretrained DRLN model from hugging face.
model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=scale)      # scale 2, 3 and 4 models available


def scale_image_for_test(image):
    """
    # Upscales images on GPU.
    # Works within the validate_drln code's context, depends on the model definition within the script.
    :param sem_file: Takes the SEM file name.
    :return: Returns the raw image, the upscaled image from DLRN and the bicubic upscaled image.
    """
    # print(image.shape)
    # Make sure image is not excessively large. If so, cut it down so its under 1024 pixels long on the longer edge.
    if image.shape[1] > 1050:
        down_scale = image.shape[1] / 1024
        image = cv2.resize(image, (int(image.shape[1] / down_scale), int(image.shape[0] / down_scale)),
                           interpolation=cv2.INTER_AREA)
    # Crop the image if it has an odd shape that would not scale up to the identical size.
    image = image[0:clean_scale_crop(image.shape[0], scale), 0:clean_scale_crop(image.shape[1], scale), :]
    # Downscale the image using area interpolation to check the upscale performance against the original.
    image_in = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)),
                          interpolation=cv2.INTER_AREA)
    # Put the image back into a PIL JPG format
    image_in = Image.fromarray(image_in)
    # Prepare the image for use in Pytorch.
    inputs = ImageLoader.load_image(image_in)
    # Upscale the image using pytorch for GPU.
    preds = model_inputs(inputs, model, cuda_gpu)

    # Convert the pytorch images to arrays.
    input_im = tensor_to_array(inputs)
    output_im = tensor_to_array(preds)

    # Use bicubic interpolation to upsample the input image to see if there is an improvement.
    input_resize = cv2.resize(input_im, (output_im.shape[1], output_im.shape[0]), interpolation=cv2.INTER_CUBIC)
    return image, output_im, input_resize


# Loop through each image and calculate to make a dataframe of SSIM, PSNR.
start = time.time()
results_list = list()
for sem_file in sem_files:
    # Load the images.
    image_a = Image.open(open(sem_file, 'rb'))
    image = asarray(image_a)

    image, output_im, input_resize = scale_image_for_test(image)
    # Compute the psnr and ssim metrics for the dlnr and bicubic upsampling images.
    dlnr_psnr = psnr(image, output_im)
    dlnr_ssim = ssim(image, output_im, win_size=3)
    bicubic_psnr = psnr(image,input_resize)
    bicubic_ssim = ssim(image, input_resize, win_size=3)

    # Append the Results to a list of dictionaries.
    results_list.append({'sem_file':sem_file,
                         'dlnr_psnr':dlnr_psnr,'dlnr_ssim':dlnr_ssim,
                         'bicubic_psnr':bicubic_psnr,'bicubic_ssim':bicubic_ssim})

# Compute and print the elapsed time.
end = time.time()
print(end-start)

# Make the dictionary list into a dataframe.
results_df = pd.DataFrame.from_dict(results_list)

# Save the results.
results_df.to_csv(r'../Results/results_list.csv')

######### Visualize the results.
results_df = pd.read_csv(r'../Results/results_list.csv')

# Calculate the difference in PSNR and SSIM for each image.
# We want to see increases in PSNR and SSIM.
results_df['psnr_diff'] = results_df['dlnr_psnr'] - results_df['bicubic_psnr']
results_df['ssim_diff'] = results_df['dlnr_ssim'] - results_df['bicubic_ssim']

results_df['Category'] = results_df.loc[:,'sem_file'].\
                             astype(str).str.rsplit('/',expand=True,n=2).\
                             loc[:,1].rename('Category')

for stat in ['psnr','ssim']:
    best_worst_df = results_df.groupby(by='Category').\
        apply(lambda x: x.sort_values(by=stat+'_diff').iloc[[0,-1],:]).drop(columns='Category')

    for cat,cat_df in best_worst_df.groupby(by='Category'):
        print(cat)
        fig, (axu,axl) = plt.subplots(2, 3)
        for ax, sem_file, bw in zip([axu,axl],cat_df['sem_file'].to_list(),['Best', 'Worst']):
            # Load the images.
            image_a = Image.open(open(sem_file, 'rb'))
            image = asarray(image_a)
            center = [int(image.shape[0]/2), int(image.shape[1]/2)]
            image = image[center[0]-100:center[0]+100,center[1]-100:center[1]+100]

            image, output_im, input_resize = scale_image_for_test(image)
            # Plot the results to compare the reference to the DLNR and bicubic up-sampling results.
            ax[0].imshow(image)
            ax[0].set_ylabel(bw)
            ax[1].imshow(output_im.astype(int).clip(0,255))
            ax[2].imshow(input_resize.astype(int).clip(0,255))
        fig.suptitle('Best and Worst '+ stat.upper() + ' Change')
        fig.savefig('../Results/'+cat+'_'+stat+'.png')
        plt.close(fig)
