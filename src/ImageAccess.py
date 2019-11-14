import cv2
import numpy as np
import pandas as pd

import os
import sys
import copy

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

    # image type constants
    ALL_TYPE = 0
    TRAIN = 1
    TEST = 2

    # image people constants
    ALL_PERSONS = 0
    AKSHAY = 1
    ANGELA = 2
    ISAAC = 3
    MARK = 4
    NABILAH = 5

    # image angle constants
    ALL_DEGREES = 0
    DEG_0 = 1
    DEG_30 = 2
    DEG_45 = 3

    # image orientation constants
    ALL_ORIENTATIONS = 0
    LEFT = 1
    RIGHT = 2

    # image color constants
    RGB = 0
    GRAYSCALE = 1

    def __init__(self):
        if ImageAccess._instance != None:
            raise Exception('Singleton. Please do not directly create instance.')
        else:    
            ImageAccess._instance = self

    def _type(self, filename):
        if filename.find("test") != -1:
            return self.TEST
        if filename.find("train") != -1:
            return self.TRAIN
        return -1

    def _person(self, filename):
        if filename.find("akshay") != -1:
            return self.AKSHAY
        if filename.find("angela") != -1:
            return self.ANGELA
        if filename.find("mark") != -1:
            return self.MARK
        if filename.find("isaac") != -1:
            return self.ISAAC
        if filename.find("nabilah") != -1:
            return self.NABILAH
        return -1
    
    def _degrees(self, filename):
        if filename.find("_0_") != -1:
            return self.DEG_0
        if filename.find("_30") != -1:
            return self.DEG_30
        if filename.find("_45") != -1:
            return self.DEG_45
        return -1

    def _orientation(self, filename):
        if filename.find("l_"):
            return self.LEFT
        if filename.find("r_"):
            return self.RIGHT

    def initialize(self):
        # obtain filenames from images directory
        dirname = os.path.join(os.path.dirname(os.getcwd()), 'images\\')
        all_files = os.listdir(dirname)
        all_image_files = [name for name in all_files if name.endswith('.jpg')]
        
        # Fastest way to initialize pandas dataframe to create rows
        # https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
        all_rows = []
        for filename in all_image_files:
            filename = os.path.join(dirname, filename)

            image_dict = {}

            image_dict['Type'] = self._type(filename)
            image_dict['Person'] = self._person(filename)
            image_dict['Degree'] = self._degrees(filename)
            image_dict['Orientation'] = self._orientation(filename)
            image_dict['Filename'] = filename
            image = cv2.imread(filename)
            image_dict['Grayscale'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            all_rows.append(image_dict)

        self._images = pd.DataFrame(all_rows, columns=('Type', 'Person', 'Degree', 'Orientation', 'Filename', 'Grayscale'))

    @staticmethod
    def  _get_instance():
        if ImageAccess._instance == None:
            ImageAccess._instance = ImageAccess()
            ImageAccess._instance.initialize()

        return ImageAccess._instance  

    @staticmethod
    def obtain_images(i_type=1, i_person=0, i_degrees=0, i_orientation=0, i_color=1):
        instance = ImageAccess._get_instance()
        current_df = instance._images

        if i_type != ImageAccess.ALL_TYPE:
            current_df = current_df.loc[current_df['Type'] == i_type]
        if i_person != ImageAccess.ALL_PERSONS:
            current_df = current_df.loc[current_df['Person'] == i_person]
        if i_degrees != ImageAccess.ALL_DEGREES:
            current_df = current_df.loc[current_df['Degree'] == i_degrees]
        if i_orientation != ImageAccess.ALL_ORIENTATIONS:
            current_df = current_df.loc[current_df['Orientation'] == i_orientation]
        
        pd.set_option('max_colwidth', 1000)
        print(current_df['Filename'])

        images = []
        if i_color == ImageAccess.RGB:
            filenames = current_df['Filename'].to_list()
            images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
            return images
        else:
            images = current_df['Grayscale'].to_numpy()
            return copy.deepcopy(images)
