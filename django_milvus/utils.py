from typing import Type

from django.db.models.base import Model
from django.db.models.query import QuerySet

from django_milvus.connection import Connection
from django_milvus.fields import MilvusField


def rebuild_index(model: Type[Model]):
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
    for instance in QuerySet(model=model).all():
        update_entry(instance)
    # conn.get_collection(model).create_index()
    for conn in connections:
        conn.flush(model)


def update_entry(instance: Model):
    for field in instance._meta.get_fields():
        if isinstance(field, MilvusField):
            conn = field.get_connection()
            conn.update_entry(instance)
