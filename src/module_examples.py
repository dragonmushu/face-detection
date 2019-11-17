import matplotlib.pyplot as plt
import cv2

from image_access import ImageAccess

"""
Module used to show use cases (examples) for different helper classes
created
"""

# ImageAccess Class use case
# Initially import ImageAccess from image_accesss module (see above)

# only call function obtain_images: ImageAccess.obtain_images(i_type, i_person, i_degrees, i_orientation, i_color)

# obtain all akshay training images as grayscale
images_1 = ImageAccess.obtain_images(i_person=ImageAccess.ANGELA)
print(images_1.shape)
# obtain all 30 degrees nabilah training images as grayscale
images_2 = ImageAccess.obtain_images(i_person=ImageAccess.NABILAH, i_degrees=ImageAccess.DEG_30)
print(images_2.shape)
# obtain all 45 degrees left oriented training images as grayscale
images_3 = ImageAccess.obtain_images(i_degrees=ImageAccess.DEG_45, i_orientation=ImageAccess.LEFT)
print(images_3.shape)
# obtain all 0 degrees akshay images as rgb
images_4 = ImageAccess.obtain_images(i_person=ImageAccess.AKSHAY, i_degrees=ImageAccess.DEG_0, i_color=ImageAccess.RGB)
print(images_4.shape)

plt.imshow(images_1[0], cmap='gray')
plt.show()
plt.imshow(images_2[0], cmap='gray')
plt.show()
plt.imshow(images_3[0], cmap='gray')
plt.show()
plt.imshow(images_4[0])
plt.show()