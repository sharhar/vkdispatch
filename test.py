import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from matplotlib import pyplot as plt

arr: np.ndarray = np.load("data/bronwyn/template_3d.npy") # np.random.rand(512, 512, 512).astype(np.float32)

#vd.initialize(log_level=vd.LogLevel.INFO)

transformed_arr = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr))).astype(np.complex64)

arr_buff = vd.Buffer((arr.shape[0], arr.shape[1]), vd.complex64)
image = vd.Image3D(arr.shape, vd.float32, 2)

@vc.shader(exec_size=lambda args: args.buff.size)
def my_shader(buff: Buff[c64], img: Img3[f32], offset: Const[v3] = [0.5, 0.5, 288.5]):
    ind = vc.global_invocation.x.copy()
    sample_coords = vc.unravel_index(ind, buff.shape).cast_to(v3)
    buff[ind] = img.sample(sample_coords + offset).xy

#print(my_shader)

image.write(transformed_arr)

my_shader(arr_buff, image)

def plot_images_and_differences(image1, image2):
    # Calculate the difference and absolute difference
    difference = image1 - image2
    abs_difference = np.abs(difference)

    print(np.sum(abs_difference))
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot the first image
    im1 = axes[0, 0].imshow(image1)
    axes[0, 0].set_title('Image 1')
    fig.colorbar(im1, ax=axes[0, 0])
    
    # Plot the second image
    im2 = axes[0, 1].imshow(image2)
    axes[0, 1].set_title('Image 2')
    fig.colorbar(im2, ax=axes[0, 1])
    
    # Plot the difference
    diff = axes[1, 0].imshow(difference)
    axes[1, 0].set_title('Difference')
    fig.colorbar(diff, ax=axes[1, 0])

    # Plot the absolute difference
    adiff = axes[1, 1].imshow(abs_difference)
    axes[1, 1].set_title('Absolute Difference')
    fig.colorbar(adiff, ax=axes[1, 1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plots
    plt.show()

#smol_buff = vd.asbuffer(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))

#sum_buff = calc_sums(smol_buff)

#print(calc_sums)

#print(sum_buff.read(0))
#print(smol_buff.read(0))

true_projection = arr.sum(axis=0)

est_proj = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(arr_buff.read(0))))

#plot_images_and_differences(transformed_arr[288, :, :].real, arr_buff.read(0).real)
plot_images_and_differences(true_projection.real, est_proj.real * 1.25)

#plt.imshow(arr[0, :, :])
#plt.show()

#ret = arr_buff.read()[0]

#plt.imshow(ret[0, :, :])
#plt.show()

#print(ret.shape)

#print(np.sum(np.abs(arr - ret)))


# whitening filter!!!!!