from typing import Type

from django.db.models.base import Model
from django.db.models.query import QuerySet

from django_milvus.connection import Connection
from django_milvus.fields import MilvusField


def rebuild_index(model: Type[Model]) -> None:
    """Removes milvus collection and recreate."""
    fields = [f for f in model._meta.get_fields() if isinstance(f, MilvusField)]
    dbnames = set([f.dbname for f in fields])
    connections = set()
    for db in dbnames:
        conn = Connection(db)
        conn.connect()
        connections.add(conn)
        if conn.has_collection(model):
            conn.remove_collection(model)
        conn.create_collection(model)
        conn.bulk_update_entries(QuerySet(model=model).all())
    for conn in connections:
        conn.flush(model)


def update_entry(instance: Model) -> None:
    fields = [f for f in instance._meta.get_fields() if isinstance(f, MilvusField)]
    dbnames = set([f.dbname for f in fields])
    for db in dbnames:
        conn = Connection(db)
        conn.connect()
        conn.update_entry(instance)
