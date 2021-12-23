import streamlit as st
import cv2
import tensorflow as tf
import os
import io
import threading
from PIL import Image
import tensorflow_addons as tfa
import numpy as np
from autocrop import Cropper





Image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])
return Image
