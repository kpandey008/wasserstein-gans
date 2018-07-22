'''
Provides the common utility operations for plotting images
'''
import os
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imageio


def cleanup(path):
    if not os.path.isdir(path):
        raise TypeError('Path provided to cleanup can only be a directory!!')
    # used to cleanup some resources
    if os.path.exists(path):
        shutil.rmtree(path)


def create_gif_from_images(src, dst, cleanup_path=[]):
    # creates a gif and stores in dst given a src
    dst_dir = dst if os.path.isdir(dst) else os.path.split(dst)[0]
    if not os.path.exists(src):
        raise OSError('No such path or directory. Did you run the optimize function for the GAN?')
    
    if not os.path.exists(dst_dir):
    # create the dst directory
        print("Destination dir not found.Creating.....")
        os.makedirs(dst_dir)
    
    print('Creating gif from the images')
    #     create the gif from the images in the source directory
    with imageio.get_writer(dst, mode='I') as writer:
    # list the images in the src ordered by time
        imageList = [os.path.join(src, image) for image in sort_by_time(src, reverse=False) if os.path.isfile(os.path.join(src, image))]
        for image in imageList:
            img = imageio.imread(image)
            writer.append_data(img)
    
    # cleanup the resources if not required
    if cleanup_path:
        for path in cleanup_path:
            cleanup(path)


def generateSamples(m, noise_dim, scale):
    # generate some random noise samples
    return np.random.normal(scale=scale, size=(m, noise_dim))


def restore_checkpoint_status(saver, sess, path):
    # check if the checkpoint exists for this experiment
    dir_path = os.path.split(path)[0] if os.path.splitext(path)[1] else path
    if not tf.train.latest_checkpoint(dir_path):
        print('No checkpoint found. Starting training.....')
        return False
    
    # else resume the training
    print('Checkpoint found for this experiment.Restoring variables.....')
    tf.reset_default_graph()
    saver.restore(sess, path)
    return True


def save_images(image_batch, img_dim_x=None, img_dim_y=None, 
                shape=None, tmp_path=None, show=False, save=False, id=None, **kwargs):

    img_shape_len = len(image_batch.shape)
    if img_shape_len != 2 and img_shape_len !=4:
        raise SyntaxError('Image shape can be either 2 dim or 4 dim with a channel last ordering for 4-dim images')

    num_channels = 1 if img_shape_len == 2 else image_batch.shape[-1]
    image_size = int(np.sqrt(image_batch.shape[1])) if img_shape_len == 2 else image_batch.shape[1]

    dim_x = img_dim_x or image_size
    dim_y = img_dim_y or image_size

   
    num_images = image_batch.shape[0]

    # calculate the grid size to display the images
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig = plt.figure(figsize=(grid_size, grid_size), **kwargs)
    img_index = 1
    
    for _ in range(grid_size):
        for _ in range(grid_size):
            
            # display the images in the plot
            fig.add_subplot(grid_size, grid_size, img_index)
            tmp_img = np.reshape(image_batch[img_index - 1], (dim_x, dim_y)) if num_channels == 1 else image_batch[img_index - 1]
            plt.imshow(tmp_img, cmap='binary')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            img_index += 1
            
    # save the image file locally
    if save and not show:
        tmp = tmp_path or os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        plt.savefig(os.path.join(tmp, '{}.png'.format(id)))
        plt.close()


def save_model_state(saver, sess, path):
    # save the model state
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    else:
        saver.save(sess, path)


def sort_by_time(folder, reverse=False):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.path.getmtime(path)
    
    return sorted(os.listdir(folder), key=getmtime, reverse=reverse)