from __future__ import annotations

from typing import TYPE_CHECKING, Type

from django.db.models import JSONField
from django.db.models.lookups import Lookup
from pymilvus.client.types import DataType

from .lookups import get_nearest_n

if TYPE_CHECKING:
    from django_milvus.connection import Connection


class MilvusField(JSONField):
    def __init__(
        self,
        dim: int,
        dtype: DataType,
        *args,
        dbname: str = "default",
        nlist: int = 1024,
        nprobe: int = 32,
        metric_type: str = "L2",
        index_type: str = "IVF_FLAT",
        **kwargs
    ):
        self.dim = dim
        self.dtype = dtype
        self.dbname = dbname
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric_type = metric_type
        self.index_type = index_type
        super().__init__(*args, **kwargs)

    def get_connection_class(self):
        from .connection import Connection

        return Connection

    def get_connection(self) -> Connection:
        return self.get_connection_class()(self.dbname)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs.update(
            {
                "dim": self.dim,
                "dtype": self.dtype,
                "dbname": self.dbname,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "metric_type": self.metric_type,
                "index_type": self.index_type,
            }
        )
        return name, path, args, kwargs

    def get_lookup(self, lookup_name: str) -> Type[Lookup] | None:
        if lookup_name.startswith("nearest"):
            try:
                return get_nearest_n(
                    int(lookup_name[8:]),
                    self.model,
                    self,
                    self.get_connection(),
                )
            except ValueError:
                raise ValueError(
                    "incorrect syntax when looking up nearby vectors: use nearest_{int}. got {lookup_name}"
                )
        else:
            raise ValueError("Not supported lookup: " + lookup_name)
