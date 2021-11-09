from typing import Any, Iterable, List, Type
from django.db.models import query
from django.db.models.query import QuerySet

import pymilvus
from django.conf import settings
from django.db.models import Model
from django.db.models.fields import CharField, Field, IntegerField
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

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
        self.build_indexes(model, collection)
        return collection

    def build_indexes(self, model: Type[Model], collection: Collection):
        for field in self.get_sorted_model_fields(model):
            collection.create_index(
                field_name=field.attname,
                index_params={
                    "metric_type": field.metric_type,
                    "index_type": field.index_type,
                    "params": {"nlist": field.nlist},
                },
            )

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

    def get_sorted_model_fields(self, model: Type[Model]) -> List[MilvusField]:
        fields = [
            f
            for f in model._meta.get_fields()
            if isinstance(f, MilvusField) and f.dbname == self.dbname
        ]
        fields.sort(key=lambda f: f.attname)
        return fields

    def get_milvus_field_schemas(self, model: Type[Model]) -> List[FieldSchema]:
        id_field = model._meta.get_field(model._meta.pk.name)
        pk_field = self.get_milvus_pk_field(id_field)
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

    def bulk_update_entries(self, queryset: QuerySet):
        self.bulk_delete_entries(queryset)
        self.bulk_insert_entries(queryset)

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

    def bulk_delete_entries(self, queryset: QuerySet):
        ...

    def insert_entry(self, instance: Model):
        collection = self.get_collection(instance._meta.model)
        collection.insert(self.get_milvus_values(instance))

    def bulk_insert_entries(self, queryset: QuerySet):
        if queryset:
            collection = self.get_collection(queryset.model)
            collection.insert(self.get_bulk_milvus_values(queryset))

    def get_milvus_values(self, instance: Model) -> List[Any]:
        fields = self.get_sorted_model_fields(instance._meta.model)
        values = [getattr(instance, f.attname) for f in fields]
        return [[instance.pk]] + [[v] for v in values]

    def get_bulk_milvus_values(self, queryset: QuerySet):
        fields = self.get_sorted_model_fields(queryset.model)
        values = queryset.values_list("pk", *[f.attname for f in fields])
        transposed = list(zip(*values))  # transpose
        return transposed

    def flush(self, model: Type[Model]):
        pymilvus.utility.get_connection().flush([self.get_collection_name(model)])
