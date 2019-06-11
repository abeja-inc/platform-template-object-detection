#    ________  ________  _______         ___  ________     
#   |\   __  \|\   __  \|\  ___ \       |\  \|\   __  \    
#   \ \  \|\  \ \  \|\ /\ \   __/|      \ \  \ \  \|\  \   
#    \ \   __  \ \   __  \ \  \_|/__  __ \ \  \ \   __  \  
#     \ \  \ \  \ \  \|\  \ \  \_|\ \|\  \\_\  \ \  \ \  \ 
#      \ \__\ \__\ \_______\ \_______\ \________\ \__\ \__\
#       \|__|\|__|\|_______|\|_______|\|________|\|__|\|__|
#                                                      


import sys, os, io

import numpy as np 
from abeja.datalake import Client as DatalakeClient
from abeja.datasets import Client as DatasetClient
from abejacli.config import ABEJA_PLATFORM_USER_ID, ABEJA_PLATFORM_TOKEN


def make_datalake_channel(name, description, organization_id):
    """
    Make Datalake channel to store some data.

    Arguments
        name : channel name [str]
        descriptino : explanation for the datalake [str]
        organization_id : your organization ID [str]
    
    Return
        datalake channel class
    
    Usage
        >>> name = "test channel"
        >>> description = "this is test datalake"
        >>> organization_id = "XXXXXXXXXXX"
        >>> channel = make_datalake_channel(name, description, organization_id)
    """

    credential = {
    'user_id': ABEJA_PLATFORM_USER_ID,
    'personal_access_token': ABEJA_PLATFORM_TOKEN
    }

    datalake_client = DatalakeClient(organization_id=organization_id, 
                                     credential=credential)
    
    channel = datalake_client.channels.create(name, description, StorageType.DATALAKE.value)

    return channel


def upload_datalake(file, channel):
    """
    Upload your file to datalake channel

    Arguments
        file : file path [str]
        channel : datalake channel 
    
    Return
        file : uploaded file
    
    Usage
        >>> channel = make_datalake_channel(name, description, organization_id)
        >>> file_paht = "./data/cat.jpg"
        >>> file = upload_datalake(file_path, channel)
    """

    file = channel.upload_file(file)

    return file


