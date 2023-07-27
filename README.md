# Lifewatch_workflow_spectral_unmixing_RGB_deep_learning
The objective of this work, is to present what is to our knowledge the first study that explores a multi-task Deep Learning approach for blind spectral unmixing using only 224x224 pixels RGB images derived from Sentinel-2 and enriched with their corresponding environmental ancillary data (topographic and climatic ancillary data) without the need to use any expensive and complex hyperspectral or multispectral data. The proposed Deep Learning model used in this study is trained in a Multi Task Learning approach (MTL) as it constitutes the most adequate machine learning method that aims to combine several information from different tasks to improve the performance of the model in each specific task, motivated by the idea that different tasks can share common feature representations. Thus, the provided model in this workflow was optimized for elaborating endmembers abundance estimation task that aims to quantify the spatial percentage covered by each LULC type within the analyzed RGB image, while being trained for other spectral unmixing related tasks that improves its accuracy in the main targeted task which is endmembers abundance estimation. The provided model here is able to give for each input (RGB image+ancillary data) the contained endmembers abundances values inside its area summarized in an output CSV file. The results can be computed for two different levels N1 and N2. These two levels reflect two land use/cover levels definitions in SIPNA land use/cover mapping campaign (Sistema de Información sobre el Patrimonio Natural de Andalucía) which aims to build an information system on the natural heritage of Andalusia in Spain (https://www.juntadeandalucia.es/medioambiente/portal/landing-page-%C3%ADndice/-/asset_publisher/zX2ouZa4r1Rf/content/sistema-de-informaci-c3-b3n-sobre-el-patrimonio-natural-de-andaluc-c3-ada-sipna-/20151). The first Level "N1" contains four high level LULC classes, whereas the second level "N2" contains ten finer level LULC classes. Thus, this model was mainly trained and validated on the region of Andalusia in Spain.

#We provided in this project, the following elements:

-"script.py" which is the main script 

-"anaconda_environment.yml" which contains the exported anaconda environment necessary to run the python script

-"input" which is the folder where the user needs to put the data inputs necessary to run the python script. This folder is supposed to have a subfolder called "RGB satellite images folder" that contains all the input RGB images, and a csv file containing their corresponding ancillary data called "Environmental ancillary data CSV file.csv". (we provided a sample of each input file inside this "input" folder)

-"output" which is the folder where the output results are stored as csv files.

-"Models" which contains the trained deep learning models for each one of the used classification levels N1 and N2.      

#To run this project sample, follow these steps:

-copy this repository in you local machine

-clone the anaconda enviroment provided in "anaconda_environment.yml" and activate it

-run the python script "script.py" 

Acknowledgemts:
This work is part of the project "Thematic Center on Mountain Ecosystem & Remote sensing, Deep learning-AI e-Services University of Granada-Sierra Nevada" (LifeWatch-2019-10-UGR-01), which has been co-funded by the Ministry of Science and Innovation through the FEDER funds from the Spanish Pluriregional Operational Program 2014-2020 (POPE), LifeWatch-ERIC action line, within the Workpackages LifeWatch-2019-10-UGR-01_WP-8, LifeWatch-2019-10-UGR-01_WP-7 and LifeWatch-2019-10-UGR-01_WP-4.
