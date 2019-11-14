import cv2
import numpy as np
import pandas as pd
import os
import copy

PATH_TO_IMAGES = "..\images\"

class ImageAccess:
    """
    Class to allow access to specific images in images folder

    Note:
        Do not directly create an instance of the class or call init function.
        Only function called should be the public static function obtain_images

    Attributes:
        _images: pandas dataframe containing image and descriptors
    """

     # singl   
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
    DEG_30 = 1
    DEG_45 = 2

    # image orientation constants
    ALL_ORIENTATIONS = 0
    LEFT = 1
    RIGHT = 2

    # image color constants
    RGB = 0
    GRAYSCALE = 1

    def __init__(self):
        if ImageAccess._instance != None:
            raise Exception('Singleton. Please use get_instance function.')
        else:    
            ImageAccess._instance = self

    def _type(filename):
        if filename.find("test"):
            return TEST
        if filename.find("train"):
            return TRAIN
        return -1

    def _person(filename):
        if filename.find("akshay") != -1:
            return AKSHAY
        if filename.find("angela") != -1:
            return ANGELA
        if filename.find("mark") != -1:
            return MARK
        if filename.find("isaac") != -1:
            return ISAAC
        if filename.find("nabilah") != -1:
            return NABILAH
        return -1
    
    def _degrees(filename):
        if filename.find("_0_") != -1:
            return DEG_0
        if filename.find("_30") != -1:
            return DEG_30
        if filename.find("_45") != -1:
            return DEG_45
        return -1

    def _orientation(filename):
        if filename.find("l_"):
            return LEFT
        if filename.find("r_"):
            return RIGHT

    def initialize(self):
        self._images = pd.DataFrame(columns=('Type', 'Person', 'Degree', 'Orientation', 'RGB', 'Grayscale'))
        
        # Fastest way to initialize pandas dataframe to create rows
        # https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
        all_filenames = os.listdir(PATH_TO_IMAGES)
        all_rows = []
        for filename in all_filenames:
            image_dict = {}

            image_dict['Type'] = _type(filename)
            image_dict['Person'] = _person(filename)
            image_dict['Degree'] = _degrees(filename)
            image_dict['Orientation'] = _orientation(filename)
            image = cv2.imread(filename)
            image_dict['RGB'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_dict['Grayscale'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            all_rows.append(image_dict)

        self._images = pd.DataFrame(all_rows, columns=('Type', 'Person', 'Degree', 'Orientation', 'RGB', 'Grayscale'))

    @staticmethod
    def  _get_instance():
        if ImageAccess._instance == None:
            ImageAccess._instance = ImageAccess()
            ImageAccess._instance.initialiaze()

        return ImageAccess._instance  

    @staticmethod
    def obtain_images(i_type=TRAIN, i_person=ALL_PERSONS, i_degrees=ALL_DEGREES, i_orientation=ALL_ORIENTATIONS, i_color=RGB):
        instance = ImageAccess._get_instance()
        current_df = instance._images

        if i_type != ALL_TYPE:
            current_df = current_df.loc[current_df['Type'] == i_type]
        if i_person != ALL_PERSONS:
            current_df = current_df.loc[current_df['Person'] == i_person]
        if i_degrees != ALL_DEGREES:
            current_df = current_df.loc[current_df['Degree'] == i_degrees]
        if i_orientation != ALL_ORIENTATIONS:
            current_df = current_df.loc[current_df['Orientation'] == i_orientation]

        images = []
        if i_color == RGB:
            images = current_df['RGB'].to_numpy()
        else:
            images = current_df['Grayscale'].to_numpy()
        
        return copy.deepcopy(images)
