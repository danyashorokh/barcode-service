
from typing import OrderedDict, List

from pydantic import BaseModel, Field


class Barcode(BaseModel):
    bbox: OrderedDict[str, int] = Field(default_factory=dict)
    value: str = None


class Response(BaseModel):
    barcodes: List[Barcode] = Field(default_factory=list)
