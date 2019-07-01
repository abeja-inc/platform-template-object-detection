import sys, os, io, json

import numpy as np 
from abeja.datalake.storage_type import StorageType
from abeja.datalake import Client as DatalakeClient
from abeja.datasets import Client as DatasetClient
from abejacli.config import ABEJA_PLATFORM_USER_ID, ABEJA_PLATFORM_TOKEN


class DataLoader:
    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.credential = {
            'user_id': ABEJA_PLATFORM_USER_ID,
            'personal_access_token': ABEJA_PLATFORM_TOKEN
        }
        self.datalake_client = DatalakeClient(organization_id=self.organization_id, 
                                              credential=self.credential)
        self.dataset_client = DatasetClient(organization_id=self.organization_id,
                                            credential=self.credential)


    def create_datalake_channel(self, name: str, description: str):
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
            >>> name = "test_hoge_foo"
            >>> description = "this is test datalake"
            >>> channel = dataloader.create_datalake_channel(name, description)
        """

        channel = self.datalake_client.channels.create(name, description, 
                                                       StorageType.DATALAKE.value)

        return channel


    def upload_datalake(self, file: str, channel):
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
            >>> channel = datalader.create_datalake_channel(name, description)
            >>> file_path = "./data/cat.jpg"
            >>> file = dataloader.upload_datalake(file_path, channel)
        """

        file = channel.upload_file(file)

        return file


    def create_dataset(self, file_path: str, dataset_name: str):
        """
        Make dataset for detection tool

        Arguments
            file_path : path to json file
                        json file should be like this form:
                            {
                                "name": "PascalVOC-2007", 
                                "type": "detection", 
                                "props": {
                                    "categories": [
                                        {
                                        "labels": [
                                            {
                                                "label_id": 0, 
                                                "label": "aeroplane"
                                            }, 
                                            {
                                                "label_id": 1, 
                                                "label": "bicycle"
                                            }, 
                                            {
                                                "label_id": 2, 
                                                "label": "bird"
                                            }, 
                                        ],
                                        "category_id": 0,
                                        "name": "PascalVOC-2007"
                                        }
                                    ]
                                }
                            }

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
            >>> file_path = "./sample.json"
            >>> dataset_name = "test_hoge"
            >>> dataset = dataloader(file_path=file_path, dataset_name=dataset_name)
        """

        with open(file_path, 'r') as f:
            dataset_props = json.load(f)
        
        dataset = self.dataset_client.datasets.create(dataset_name, dataset_props['type'], dataset_props['props'])

        return dataset


    def download_dataset(self, dataset_id: str, max_num=None, prefetch=False):
        """
        Download your favorite dataset

        Arguments
            dataset_id : dataset id for ABEJA Platform [str]
                         ex) dataset_id = "1788652111540"
            max_num : the max number of dataset itemsd
            prefetch : whether or not to use cache to download dataset
        
        Return
            dataset_item [List]
        
        Usage
            >>> organization_id = "XXXXXXXXXXXX"
            >>> dataloader = DataLoader(organization_id)
            >>> dataset = dataloader.download_dataset(dataset_id)
        """

        dataset = self.dataset_client.get_dataset(dataset_id)

        return dataset

