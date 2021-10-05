import re
import os
import numpy as np
import torch

def get_path_list(top_dir,file_regex):
    """

    :param top_dir: Directory path to search in.
    :param file_regex: Key regular expression phrase to search for.
    :return: A list of full file paths to every matching file in the directory.
    """
    relevant_list = list()
    reg = re.compile(file_regex)
    for dir_path, dirs, files in os.walk(top_dir):
        for filename in files:
            f_name = os.path.join(dir_path,filename)
            if reg.search(f_name):
                relevant_list.append(f_name)
    return relevant_list

def tensor_to_array(pred):
    """

    :param pred: Tensorflow image array.
    :return: Numpy image array.
    """
    pred = pred.detach().numpy()
    pred = pred[0].transpose((1, 2, 0)) * 255.0
    return pred

def clean_scale_crop(dim,scale):
    """

    :param dim: Single dimension of an image.
    :param scale: Scaling factor for image.
    :return: # Returns the nearest whole number to the original dimension that can be upscaled and downscaled.
    """
    return int(np.floor(dim / scale)*scale)

def model_inputs(input_data,model,device):
    """

    :param input_data: image data tensor
    :param model: pretrained model.
    :param device: gpu device variable for pytorch.
    :return: returns the predicted outputs from the model.
    """
    model = model.to(device=device)
    with torch.no_grad():
        input_data = input_data.to(device=device)
        preds = model(input_data)
        preds_out = preds.cpu()
    del preds, model, input_data
    return preds_out