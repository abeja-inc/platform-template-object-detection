#                                                                           
#          db         88888888ba   88888888888         88        db         
#         d88b        88      "8b  88                  88       d88b        
#        d8'`8b       88      ,8P  88                  88      d8'`8b       
#       d8'  `8b      88aaaaaa8P'  88aaaaa             88     d8'  `8b      
#      d8YaaaaY8b     88""""""8b,  88"""""             88    d8YaaaaY8b     
#     d8""""""""8b    88      `8b  88                  88   d8""""""""8b    
#    d8'        `8b   88      a8P  88          88,   ,d88  d8'        `8b   
#   d8'          `8b  88888888P"   88888888888  "Y8888P"  d8'          `8b  
#


from abeja.datalake import Client as DatalakeClient
from abeja.datalake.storage_type import StorageType
from abejacli.config import (
    ABEJA_PLATFORM_USER_ID, ABEJA_PLATFORM_TOKEN
)


def make_datalake_channel(name, description, organization_id):
    """
    Make Datalake channel to store some data.

    Parameters
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

    