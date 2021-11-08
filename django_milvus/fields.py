from __future__ import annotations
from typing import Any, List, Type
from django.db.models import JSONField, Field, QuerySet
from django.db.models.base import Model
from django.db.models.lookups import Lookup, Transform
from pymilvus.client.types import DataType

from .lookups import get_nearest_n


class MilvusField(JSONField):
    def __init__(
        self, dim: int, dtype: DataType, *args, dbname: str = "default", **kwargs
    ):
        self.dim = dim
        self.dtype = dtype
        self.dbname = dbname
        super().__init__(*args, **kwargs)

    def get_connection_class(self):
        from .connection import Connection

        return Connection

    def get_connection(self):
        return self.get_connection_class()(self.dbname)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs.update(
            {
                "dim": self.dim,
                "dtype": self.dtype,
                "dbname": self.dbname,
            }
        )
        return name, path, args, kwargs

    def get_lookup(self, lookup_name: str) -> Type[Lookup] | None:
        if lookup_name.startswith("nearest"):
            try:
                return get_nearest_n(
                    int(lookup_name[8:]),
                    self.model,
                    self.attname,
                    self.get_connection(),
                )
            except ValueError:
                raise ValueError(
                    "incorrect syntax when looking up nearby vectors: use nearest_{int}. got {lookup_name}"
                )
        else:
            raise ValueError("Not supported lookup: " + lookup_name)


def rebuild_index(model: Type[Model]):
    """Removes milvus collection and recreate."""
    dbnames = set()
    for field in model._meta.get_fields():
        if isinstance(field, MilvusField):
            if field.dbname in dbnames:
                continue
            dbnames.add(field.dbname)
            conn = field.get_connection()
            conn.connect()
            if conn.has_collection(model):
                conn.remove_collection(model)
            conn.create_collection(model)
    for instance in QuerySet(model=model).all():
        update_entry(instance)


def update_entry(instance: Model):
    for field in instance._meta.get_fields():
        if isinstance(field, MilvusField):
            conn = field.get_connection()
            conn.update_entry(instance)
