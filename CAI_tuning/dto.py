from typing import Union
from pydantic import BaseModel

class ReqQuery(BaseModel):
    model_id: Union[str, None] = None 
    gpus: Union[str, None] = None 


    