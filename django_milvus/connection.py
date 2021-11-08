from typing import TYPE_CHECKING, Any, List, Type
from django.db import models
from django.db.models.base import Model
from django.db.models.fields import (
    AutoField,
    BigAutoField,
    CharField,
    Field,
    IntegerField,
)
from django.conf import settings
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
import pymilvus
from django_milvus.fields import MilvusField


class Connection:
    def __init__(self, dbname: str):
        self.dbname = dbname

    def connect(self):
        MILVUS = settings.MILVUS
        config = MILVUS["DATABASES"][self.dbname]
        HOST = config["HOST"]
        PORT = config["PORT"]
        pymilvus.connections.connect(self.dbname, host=HOST, port=str(PORT))

    def has_collection(self, model: Type[Model]) -> Collection:
        return pymilvus.utility.get_connection().has_collection(
            self.get_collection_name(model)
        )

    def get_collection(self, model: Type[Model]) -> pymilvus.Collection:
        return pymilvus.Collection(name=self.get_collection_name(model))

    def get_collection_name(self, model: Type[Model]) -> str:
        return model.__name__.lower()

    def create_collection(self, model: Type[Model]) -> Collection:
        model_name = model.__name__.lower()
        schema = CollectionSchema(
            fields=self.get_milvus_field_schemas(model),
            description=f"collection for {model_name}",
        )
        collection = Collection(
            name=self.get_collection_name(model),
            schema=schema,
            using=self.dbname,
            shards_num=2,
        )
        return collection

    def remove_collection(self, model: Type[Model]):
        self.get_collection(model).drop()

    def get_milvus_pk_field(self, primary_key: Field) -> FieldSchema:
        if isinstance(primary_key, CharField):
            return FieldSchema(
                name=primary_key.attname,
                dtype=DataType.STRING,
                is_primary=True,
            )
        elif isinstance(primary_key, IntegerField):
            return FieldSchema(
                name=primary_key.attname,
                dtype=DataType.INT64,
                is_primary=True,
            )
        else:
            raise NotImplementedError(
                f"Not supported: {primary_key.__class__}({primary_key})"
            )

    # def get_milvus_soft_delete_field(self):
    #     return FieldSchema(
    #         name="_deleted",
    #         dtype=DataType.BOOL,
    #     )

    def get_sorted_model_fields(self, model: Type[Model]) -> List[MilvusField]:
        fields = [
            f
            for f in model._meta.get_fields()
            if isinstance(f, MilvusField) and f.dbname == self.dbname
        ]
        fields.sort(key=lambda f: f.attname)
        return fields

    def get_milvus_field_schemas(self, model: Type[Model]) -> List[FieldSchema]:
        id_field = model._meta.get_field("id")
        pk_field = self.get_milvus_pk_field(id_field)
        # soft_delete_field = self.get_milvus_soft_delete_field()
        return [pk_field] + [
            FieldSchema(
                name=f.attname,
                dtype=f.dtype,
                dim=f.dim,
            )
            for f in self.get_sorted_model_fields(model)
        ]

    def update_entry(self, instance: Model):
        self.check_schema(instance._meta.model)
        self.delete_entry(instance)
        self.insert_entry(instance)

    def check_schema(self, model: Type[Model]):
        current_fields = self.get_milvus_field_schemas(model)
        collection_fields: List[FieldSchema] = self.get_collection(model).schema.fields
        expected = [f.attname for f in current_fields]
        actual = [f.name for f in collection_fields]
        if expected != expected:
            raise ValueError(f"Schema mismatch: {expected=} {actual=}")

    def delete_entry(self, instance: Model):
        # pk = instance.pk
        # collection = self.get_collection(instance._meta.model)
        # collection.delete(pk)
        ...

    def insert_entry(self, instance: Model):
        collection = self.get_collection(instance._meta.model)
        collection.insert(self.get_milvus_values(instance))

    def get_milvus_values(self, instance: Model) -> List[Any]:
        fields = self.get_sorted_model_fields(instance._meta.model)
        values = [getattr(instance, f.attname) for f in fields]
        return [[instance.pk]] + [[v] for v in values]
