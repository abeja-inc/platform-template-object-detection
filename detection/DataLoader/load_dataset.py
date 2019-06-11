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


class DataLoader:
    def __init__(self, organizatino_id):
        self.organization_id = organization_id
        self.credential = {
            'user_id': ABEJA_PLATFORM_USER_ID,
            'personal_access_token': ABEJA_PLATFORM_TOKEN
        }

    def create_datalake_channel(self, name, description):
        """
        Make Datalake channel to store some data.

        Arguments
            name : channel name [str]
            descriptino : explanation for the datalake [str]
            organization_id : your organization ID [str]

        Return
            datalake channel class

        Usage
            >>> organization_id = "XXXXXXXXXXX"
            >>> dataloader = DataLoader(organization_id)
            >>> name = "test channel"
            >>> description = "this is test datalake"
            >>> channel = dataloader.make_datalake_channel(name, description)
        """

        datalake_client = DatalakeClient(organization_id=self.organization_id, 
                                         credential=self.credential)

        channel = datalake_client.channels.create(name, description, StorageType.DATALAKE.value)

        return channel


    def upload_datalake(self, file, channel):
        """
        Upload your file to datalake channel

        Arguments
            file : file path [str]
            channel : datalake channel 

        Return
            file : uploaded file

        Usage
            >>> organization_id = "XXXXXXXXXXX"
            >>> dataloader = DataLoader(organization_id)
            >>> channel = dataloader.make_datalake_channel(name, description)
            >>> file_paht = "./data/cat.jpg"
            >>> file = dataloader.upload_datalake(file_path, channel)
        """

        file = channel.upload_file(file)

        return file


    def create_dataset(self, label_lst, category_lst, dataset_name, task_type):
        """
        Make dataset for detection tool

        Arguments
            label_lst : Label name for each category [List[str]]
                        If you want to attach label for animal category and human category,
                        ex) label_lst = [["dog", "cat"], ["man", "woman"]]
            category_lst : name for each category [List[str]]
                        ex) category_lst = ["animal", "human"]
            dataset_name : your dataset name for ABEJA PLATFORM [str]
            type : name of this task [str]
                   ex) type = "detection"
                       type = "classification"
                       type = "custom"
            organization_id : your organization_id [str]

        Return
            your uploaded dataset

        Usage
            >>> organization_id = "XXXXXXXXXXX"
            >>> dataloader = DataLoader(organization_id)
            >>> label_lst = [["dog", "cat"], ["man", "woman"]]
            >>> ategory_lst = ["animal", "human"]
            >>> dataset = dataloader.make_dataset_detection(label_lst, category_lst)
        """

        # assert same length
        if len(label_lst) != len(category_lst):
            print("Invalid arguments: label_lst and category_lst should be sama length.")
            sys.exit()

        categories = list()
        for category_id, category_name in enumerate(category_lst):
            category = dict()
            labels = list()
            for label_id, label_name in enumerate(label_lst[category_id]):
                label = dict()
                label["label_id"] = label_id
                label["label"] = label_name
                labels.append(label)
            category["labels"] = labels
            category["category_id"] = category_id
            category["name"] = category_name
            categories.append(category)

        props = {"categories": categories}    

        dataset = datasets_client.datasets.create(name=dataset_name, type=task_type, props=props)

        return dataset


    def upload_dataset(self, data_mask, file, channel):
        """
        Upload file on dataset with annotation

        Arguments
            data_mask : label data of input image
                        ex) classification -> {"category_id": 0, 
                                               "label_id": 1}
                            detection -> {"category_id": 0,
                                          "label_id": 2,
                                          "rect": {'xmin': 200, 
                                                   'ymin': 0, 
                                                   'xmax': 1000, 
                                                   'ymax': 900}}
            file : your file path to upload [str]
            channel : datalake channel
        
        Return
            dataset_item : uploaded file with annotation
        
        Usage
            >>> organization_id = "XXXXXXXXX"
            >>> dataloader = DataLoader(organization_id)
            >>> datalake_name = "hogehoge lake"
            >>> datalake_description = "this is hogehoge lake test"
            >>> channel = dataloader.create_datalake_channel(datalake_name, datalake_description)
            >>> file_path = "./tmp/foo.jpg"
            >>> file = dataloader.upload_datalake(file_path, channel)
            >>> data_mask = {"category_id": 0,
                             "label_id": 2,
                             "rect": {'xmin': 200, 
                                      'ymin': 0, 
                                      'xmax': 1000, 
                                      'ymax': 900}}
        """

        source_data = [{
                'data_type': 'image/jpeg',
                'data_uri': 'datalake://{}/{}'.format(channel.channel_id, file.file_id),
               }
        ]

        attributes = {'detection': [data_mask]}

        dataset_item = dataset.dataset_items.create(source_data=source_data, attributes=attributes)

        return dataset_item