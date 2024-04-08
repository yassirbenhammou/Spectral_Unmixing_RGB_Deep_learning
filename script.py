import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")



##################################################### Paths to the input data ##############################################

Images_absolute_path='./input/RGB satellite images folder/'
merged_dataframe=pd.read_csv('./input/Environmental ancillary data CSV file.csv')


########################################################################################################################
################################################## Preprocessing #######################################################
########################################################################################################################


##################################################### Check if the input data is valid ##############################################
print("starting...")
print("Input data verification....")

RGB_Images=[]
#getting all the provided RGB images in a list
for filename in os.listdir(Images_absolute_path):
    f = os.path.join(Images_absolute_path, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith(".jpg"):
        RGB_Images.append(filename)

#verifying images size
for i in RGB_Images:
  image_full_path = os.path.join(Images_absolute_path, i)
  im = cv2.imread(image_full_path)
  h, w, c = im.shape
  if(w != 224 or h != 224 or c != 3):
    message_error='There is some RGB image in your provided input folder that does not have the required size of 224x224x3'
    print(message_error)
    exit()

#verifying that the RGB images and CSV ancillary data has the same size 
if (len(RGB_Images) != len(merged_dataframe)):
  message_error='There is not as much RGB image in your provided input folder as entries in your csv file'
  print(message_error)
  exit()

#verifying that some RGB images in th required format have been provided 
if (len(RGB_Images)==0):
    message_error='There is no valid RGB image in your provided input folder, Please ensure that your folder contains RGB images in .jpg format'
    print(message_error)
    exit()

#verifying that all RGB images in the input folder have their corresponding ancillary data rows in the input csv file
for img in RGB_Images:
  if (img not in merged_dataframe['filename'].values.tolist()):
    print('Some of your RGB images in the input folder does not have their corresponding ancillary data rows in the input csv file')
    exit()

#verifying that all rows in the csv ancillary data have their corresponding RGB images
for i,row in merged_dataframe.iterrows():
  if(row['filename'] not in RGB_Images):
    message_error='Some of your csv file entries does not have their corresponding RGB images, Please ensure that all your csv rows have their correspondig RGB images in the provided folder'
    print(message_error)
    exit()

#verifying that the csv ancillary data columns are valid
provided_columns=merged_dataframe.columns.values.tolist()
if (not set(['filename','evapotranspiration', 'precipitation', 'elevation', 'slope', 'temp_ave', 'temp_max', 'temp_min']).issubset(provided_columns)):
  message_error="Your csv files columns are not valid, please ensure that your provided columns contains the following: ['filename','evapotranspiration', 'precipitation', 'elevation', 'slope', 'temp_ave', 'temp_max', 'temp_min']"
  print(message_error)
  exit()

#verifying the data type provided in each column of the csv file
data_types = merged_dataframe.dtypes
condition= data_types['filename']=='object'  and data_types['evapotranspiration']=='float64' and data_types['precipitation']=='float64' and data_types['elevation']=='float64' and data_types['slope']=='float64' and data_types['temp_ave']=='float64' and data_types['temp_max']=='float64' and data_types['temp_min']=='float64'
if(not condition):
  message_error='the data types provided in the columns of your csv file are not valid, please ensure that your csv file columns have valid row values'
  exit()


print("Input data verification done: Your data is valid ")


#####################################################Useful function to get X_pic, X_metadata from a DataFrame##############################################
def get_X_y(df):

  X_pic, X_metadata = [], []

  for path_pic in df['NPZ_Path']:
    loaded_npz = np.load(path_pic)

    pic = loaded_npz['pic']
    X_pic.append(pic)

    metadata = loaded_npz['metadata']
    X_metadata.append(metadata)
    
  X_pic, X_metadata = np.array(X_pic), np.array(X_metadata)

  return (X_pic, X_metadata)


#############################Store Data in Compressed NumPy array files (.NPZs)###################################
print("Creation of compressed NPZ files to group the images and their ancillary data...")
npz_paths = []

for i,row in merged_dataframe.iterrows():
  picture_path = Images_absolute_path+row['filename']
  npz_path = os.path.splitext(picture_path)[0]+'.npz'
  npz_paths.append(npz_path)

  pic_bgr_arr = cv2.imread(picture_path)
  pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

  evapotranspiration, precipitation, elevation, slope = row['evapotranspiration'], row['precipitation'], row['elevation'], row['slope']
  temp_ave, temp_max, temp_min = row['temp_ave'], row['temp_max'], row['temp_min']
  metadata = np.array([evapotranspiration, precipitation, elevation, slope, temp_ave, temp_max, temp_min])
  np.savez_compressed(npz_path, pic=pic_rgb_arr, metadata=metadata)
print("NPZ files creation done")
merged_dataframe['NPZ_Path'] = pd.Series(npz_paths)


########################################################################################################################
#################################################### Inference #########################################################
########################################################################################################################

#############################Drop metadata columns since we have that stored in the NPZ file #############################
merged_dataframe=merged_dataframe.drop(['evapotranspiration', 'precipitation', 'elevation', 'slope', 'temp_ave', 'temp_max', 'temp_min'], axis=1)

########################################## Get the Data ##########################################################
(X_test_pic, X_test_metadata) = get_X_y(merged_dataframe)

########################################## Get the test Data ##########################################################
level = input("Please enter the clessification level you want to use. To use level N1, write 'N1' and press Enter. To use level N2, write 'N2' and press Enter. (Or press enter for default 'N1' level): ") or "N1"
print('Selected level is: '+level)

########################################### Load the model and compute the results ############################################################
print("Starting the model inference using the input data...")
if(level=='N1'):
  #load the model
  model = load_model("./Models/ML_MIMO_"+level+"_S2G")
  df_results_N1 = pd.DataFrame(columns = ['filename', 'Artificial','Agricultural lands','Terrestrial wildlands','Wetlands and riverine forests'])
  #make predictions on the data 
  y_test_predicted, y_ml_predicted = model.predict([X_test_pic, X_test_metadata])
  for r in range(y_test_predicted.shape[0]):
    df_results_N1 = df_results_N1.append({'filename' :os.path.splitext(merged_dataframe['filename'].values.tolist()[r])[0]+'.jpg', 'Artificial' :y_test_predicted[r][0]*100, 'Agricultural lands' :y_test_predicted[r][1]*100, 'Terrestrial wildlands' :y_test_predicted[r][2]*100, 'Wetlands and riverine forests' : y_test_predicted[r][3]*100}, ignore_index = True)
  df_results_N1.to_csv('./output/Abundance estimation results at level '+level+'.csv', index=False)


if(level=='N2'):
  #load the model
  model = load_model("./Models/ML_MIMO_"+level+"_S2G")
  df_results_N2 = pd.DataFrame(columns = ['filename', 'Artificial','Annual croplands','Greenhouses','Woody croplands','Combinations of croplands and natural vegetation','Grasslands and Grasslands with trees','Shrubland and Shrublands with trees','Forests','Barelands','Wetlands'])
  #make predictions on the data
  y_test_predicted, y_ml_predicted = model.predict([X_test_pic, X_test_metadata])
  for r in range(y_test_predicted.shape[0]):
    df_results_N2 = df_results_N2.append({'filename' :os.path.splitext(merged_dataframe['filename'].values.tolist()[r])[0]+'.jpg', 'Artificial' :y_test_predicted[r][0]*100, 'Annual croplands' :y_test_predicted[r][1]*100, 'Greenhouses' :y_test_predicted[r][2]*100, 'Woody croplands' : y_test_predicted[r][3]*100, 'Combinations of croplands and natural vegetation' :y_test_predicted[r][4]*100, 'Grasslands and Grasslands with trees' :y_test_predicted[r][5]*100, 'Shrubland and Shrublands with trees' :y_test_predicted[r][6]*100, 'Forests' :y_test_predicted[r][7]*100, 'Barelands' :y_test_predicted[r][8]*100, 'Wetlands' :y_test_predicted[r][9]*100}, ignore_index = True)
  df_results_N2.to_csv('./output/Abundance estimation results at level '+level+'.csv', index=False)

print("the model inference using the input data is done and the results are saved in the output folder")

