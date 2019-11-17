import cv2
import numpy as np
import pandas as pd
import pickle

import os
import copy

from helper import *

# image size
IMAGE_COMPRESSION_SIZE = 256

def type_from_filename(filename):
        if filename.find("test") != -1:
            return DataType.TEST
        if filename.find("train") != -1:
            return DataType.TRAIN
        return -1

def person_from_filename(filename):
    if filename.find("akshay") != -1:
        return Label.AKSHAY
    if filename.find("angela") != -1:
        return Label.ANGELA
    if filename.find("mark") != -1:
        return Label.MARK
    if filename.find("isaac") != -1:
        return Label.ISAAC
    if filename.find("nabilah") != -1:
        return Label.NABILAH
    return -1

def degrees_from_filename(filename):
    if filename.find("_0_") != -1:
        return Angle.DEG_0
    if filename.find("_30") != -1:
        return Angle.DEG_30
    if filename.find("_45") != -1:
        return Angle.DEG_45
    return -1

def orientation_from_filename(filename):
    if filename.find("l_") != -1:
        return Orientation.LEFT
    if filename.find("r_") != -1:
        return Orientation.RIGHT

def preprocess_images(images_directory):
    # obtain filenames from images directory
    all_files = os.listdir(images_directory)
    all_image_files = [name for name in all_files if name.endswith('.jpg')]
    
    # Fastest way to initialize pandas dataframe to create rows
    # https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
    all_rows = []
    for filename in all_image_files:
        filename = os.path.join(images_directory, filename)

        image_dict = {}

        image_dict['Type'] = type_from_filename(filename)
        image_dict['Person'] = person_from_filename(filename)
        image_dict['Degree'] = degrees_from_filename(filename)
        image_dict['Orientation'] = orientation_from_filename(filename)
        image_dict['Filename'] = filename

        image = cv2.imread(filename)
        image = cv2.resize(image, (IMAGE_COMPRESSION_SIZE, IMAGE_COMPRESSION_SIZE))
        image_dict['RGB'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_dict['Grayscale'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_rows.append(image_dict)

    # create dataframe of images
    images = pd.DataFrame(all_rows, columns=('Type', 'Person', 'Degree', 'Orientation', 'Filename', 'Grayscale', 'RGB'))

    # pickle and save images
    images.to_pickle('data.pkl')

    return images

class ImageAccess:
    """
    Class to allow access to specific images in images folder

    Note:
        Do not directly create an instance of the class or call init function.
        Only function called should be the public static function obtain_images

    Attributes:
        _images: pandas dataframe containing image and descriptors
    """

     # singleton static instance   
    _instance = None

    def __init__(self):
        if ImageAccess._instance != None:
            raise Exception('Singleton. Please do not directly create instance.')
        else:    
            ImageAccess._instance = self

    def initialize(self):
        if os.path.exists('data.pkl'):
            self._images = df = pd.read_pickle('data.pkl')
        else:
            dirname = os.path.join(os.path.dirname(os.getcwd()), 'images')
            self._images = preprocess_images(dirname)
        
    @staticmethod
    def  _get_instance():
        if ImageAccess._instance == None:
            ImageAccess._instance = ImageAccess()
            ImageAccess._instance.initialize()

        return ImageAccess._instance

    @staticmethod
    def _filter_df(i_type, i_person, i_degrees, i_orientation, i_color):
        instance = ImageAccess._get_instance()
        current_df = instance._images

        if i_type != DataType.ALL_TYPE:
            current_df = current_df.loc[current_df['Type'] == i_type]
        if i_person != Label.ALL_PERSONS:
            current_df = current_df.loc[current_df['Person'] == i_person]
        if i_degrees != Angle.ALL_DEGREES:
            current_df = current_df.loc[current_df['Degree'] == i_degrees]
        if i_orientation != Orientation.ALL_ORIENTATIONS:
            current_df = current_df.loc[current_df['Orientation'] == i_orientation]
        
        return current_df

    @staticmethod
    def obtain_images(i_type=Type.TRAIN, i_person=Label.ALL_PERSONS, i_degrees=Angle.ALL_DEGREES, i_orientation=Orientation.ALL_ORIENTATIONS, i_color=Color.RGB):
        """
        Function to obtain images you want to use for processing or testing 

        Note:
            if obtaining rgb images be careful not to choose too many so you don't run out of memory

        Args:
            i_type: type of image (see helper.py). Default set to ImageAccess.ALL_TYPES
            i_person: person you want (see helper.py). Default set to ImageAccess.ALL_PERSONS
            i_degrees: image angle desired (see helper.py). Default set to ImageAccess.ALL_DEGREES
            i_orientation: desired orientation (see helper.py). Default set to ImageAccess.ALL_PERSONS
            i_color: image color scheme (see helper.py). Default set to ImageAccess.Grayscale

        Returns:
            np array of images filtered from data frame
        """

        current_df = ImageAccess._filter_df(i_type, i_person, i_degrees, i_orientation, i_color)

        if i_color == Color.RGB:
            images = current_df['RGB'].to_numpy()
            return images
        else:
            images = current_df['Grayscale'].to_numpy()
            return images

    @staticmethod
    def obtain_labeled_data(i_type=Type.TRAIN, i_person=Label.ALL_PERSONS, i_degrees=Angle.ALL_DEGREES, i_orientation=Orientation.ALL_ORIENTATIONS, i_color=Color.RGB):
        """
        Function to obtain images with person label (see helper.py) you want to use for processing or testing 

        Note:
            if obtaining rgb images be careful not to choose too many so you don't run out of memory

        Args:
            i_type: type of image (see helper.py). Default set to ImageAccess.ALL_TYPES
            i_person: person you want (see helper.py). Default set to ImageAccess.ALL_PERSONS
            i_degrees: image angle desired (see helper.py). Default set to ImageAccess.ALL_DEGREES
            i_orientation: desired orientation (see helper.py). Default set to ImageAccess.ALL_PERSONS
            i_color: image color scheme (see helper.py). Default set to ImageAccess.Grayscale

        Returns:
            np array of images filtered from data frame
        """

        current_df = ImageAccess._filter_df(i_type, i_person, i_degrees, i_orientation, i_color)

        if i_color == Color.RGB:
            images = current_df['RGB'].to_numpy()
            labels = current_df['Person'].to_numpy()
            return images, labels
        else:
            images = current_df['Grayscale'].to_numpy()
            labels = current_df['Person'].to_numpy()
            return images, labels