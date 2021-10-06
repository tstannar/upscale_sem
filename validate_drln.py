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


# Loop through each image and calculate to make a dataframe of SSIM, PSNR.
start = time.time()
results_list = list()
for sem_file in sem_files:
    # Load the images.
    image_a = Image.open(open(sem_file,'rb'))
    image = asarray(image_a)
    # print(image.shape)
    # Make sure image is not excessively large. If so, cut it down so its under 1024 pixels long on the longer edge.
    if image.shape[1]>1050:
        down_scale = image.shape[1]/1024
        image = cv2.resize(image, (int(image.shape[1] / down_scale), int(image.shape[0] / down_scale)),
                              interpolation=cv2.INTER_AREA)
    # Crop the image if it has an odd shape that would not scale up to the identical size.
    image = image[0:clean_scale_crop(image.shape[0],scale), 0:clean_scale_crop(image.shape[1],scale),:]
    # Downscale the image using area interpolation to check the upscale performance against the original.
    image_in = cv2.resize(image,(int(image.shape[1]/scale),int(image.shape[0]/scale)),interpolation = cv2.INTER_AREA)
    # Put the image back into a PIL JPG format
    image_in = Image.fromarray(image_in)
    # Prepare the image for use in Pytorch.
    inputs = ImageLoader.load_image(image_in)
    # Upscale the image using pytorch for GPU.
    preds = model_inputs(inputs,model,cuda_gpu)

    # Convert the pytorch images to arrays.
    input_im = tensor_to_array(inputs)
    output_im = tensor_to_array(preds)

    # Use bicubic interpolation to upsample the input image to see if there is an improvement.
    input_resize = cv2.resize(input_im, (output_im.shape[1], output_im.shape[0]), interpolation = cv2.INTER_CUBIC)

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

# Make the dictionary list into a dataframe and save the results.
results_df = pd.DataFrame.from_dict(results_list)
results_df.to_csv(r'../Results/results_list.csv')

# Plot the results to compare the reference to the DLNR and bicubic upsampling results.
# fig, (ax1, ax2,ax3 ) = plt.subplots(1,3)
# ax1.imshow(image)
# ax2.imshow(output_im.astype(int).clip(0,255))
# ax3.imshow(input_resize.astype(int).clip(0,255))

