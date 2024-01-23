#!/usr/bin/env python
# coding: utf-8

# # Downloading mitocheck traiing data
#
# In this notebook, we are acquiring the MitoCheck dataset from the [MitoCheck Data repository](https://github.com/WayScience/mitocheck_data).
# We will specifically download the [training dataset](https://github.com/wayscience/mitocheck_data/blob/main/3.normalize_data/normalized_data/training_data.csv.gz) and the control dataset.
#
# The downloaded datasets will be saved in the `./raw` folder, designated for storing all raw datasets within this repository.
#
# Data information:
#
# - negative_control_data/ : Nuclei features from negative control cells transfected with scrambled siRNA.
# - positive_control_data/ : Nuclei features from positive control cells transfected with siRNA targeting genes used during mitosis (INCENP, KIF11, COPB1).
# - training_data/ : Nuclei features from cells manually labeled with a phenotypic class by MitoCheck Consortium.
#
# More information can be found [here](https://zenodo.org/records/7967386)

# In[18]:


import logging
import pathlib
import zipfile

import requests

# In[21]:


# downloading the data
log_file_name = pathlib.Path("download_log.log")
chunk_size_download = 81920
control_url = ""
train_url = "https://github.com/wayscience/mitocheck_data/raw/main/3.normalize_data/normalized_data/training_data.csv.gz"
control_url = "https://zenodo.org/records/7967386/files/3.normalize_data__normalized_data.zip?download=1"

# output paths
raw_dir = pathlib.Path("./raw/").resolve(strict=True)
train_outname = train_url.split("/")[-1]
control_out_path = raw_dir / "3.normalized_data.zip"
train_out_path = raw_dir / train_outname
control_unzip_path = raw_dir / "normalized_data"


# setting up logger
logging.basicConfig(
    filename=log_file_name,
    level=logging.DEBUG,
    format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
)


# In[16]:


# downloading training data using requests
logging.info(f"Downloading trianing set data from: {train_url}")
with requests.get(train_url, stream=True) as r:
    # raise error if the there's an error
    r.raise_for_status()
    logging.info(f"Downloading training dataset: {r.headers.get('Content-Length')}MB")

    # creating a file to write the downloaded contents in chunks
    with open(train_out_path, mode="wb") as out_file:
        for chunk in r.iter_content(chunk_size=chunk_size_download):
            out_file.write(chunk)


# In[17]:


# downloading control data using requests
logging.info(f"Downloading control dataset from: {train_url}")
with requests.get(control_url, stream=True) as r:
    # raise error if the there's an error
    r.raise_for_status()
    logging.info(f"Downloading control dataset: {r.headers.get('Content-Length')}MB")

    # creating a file to write the downloaded contents in chunks
    with open(control_out_path, mode="wb") as out_file:
        for chunk in r.iter_content(chunk_size=1024**2):
            out_file.write(chunk)


# In[25]:


# next is to unzip the control dataset inside the raw data folder
logging.info(f"Unzipping control dataset into: {raw_dir}")
with zipfile.ZipFile(control_out_path, mode="r") as zip_ref:
    zip_ref.extractall(raw_dir)
