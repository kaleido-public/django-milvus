from typing import Any, List, Type, TypeVar
from uuid import UUID

import pymilvus
from django.conf import settings
from django.db.models import Model
from django.db.models.query import QuerySet
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

from django_milvus.fields import MilvusField

Int64 = int
Int128 = int


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

    def build_indexes(self, model: Type[Model], collection: Collection) -> None:
        for field in self.get_sorted_model_fields(model):
            collection.create_index(
                field_name=field.attname,
                index_params={
                    "metric_type": field.metric_type,
                    "index_type": field.index_type,
                    "params": {"nlist": field.nlist},
                },
            )

    def remove_collection(self, model: Type[Model]) -> None:
        self.get_collection(model).drop()

    def get_sorted_model_fields(self, model: Type[Model]) -> List[MilvusField]:
        fields = [
            f
            for f in model._meta.get_fields()
            if isinstance(f, MilvusField) and f.dbname == self.dbname
        ]
        fields.sort(key=lambda f: f.attname)
        return fields

    def get_milvus_field_schemas(self, model: Type[Model]) -> List[FieldSchema]:
        """
        We are using this layout: [
            id: int64,
            django_pk_high: int8, (this is final 2 bits)
            django_pk_mid: int64, (this is the next 63 bits)
            django_pk_low: int64, (this is the first 63 bits)
        ] followed by sorted [
            django_field_name: vector_field,
            for each declared MilvusField on the Django model.
        ]

        This workaround is because milvus doesn't support the String data type yet,
        and we need a way to store the UUID primary key, which is 128 bits.


        The uuid is ((django_pk_high & mask_high) << 126) + ((django_pk_mid & mask_mid) << 63) + (django_pk_low & mask_low)
        mask_high is 0b11
        mask_mid and mask_low are (1 << 63) - 1 (every where 1 except the left most bit)
        """
        return [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
            ),
            FieldSchema(
                name="django_pk_high",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="django_pk_mid",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="django_pk_low",
                dtype=DataType.INT64,
            ),
        ] + [
            FieldSchema(
                name=f.attname,  # this is the field name on the django model
                dtype=f.dtype,  #  user specifies the type on the MilvusField on the Django model
                dim=f.dim,
            )
            for f in self.get_sorted_model_fields(model)
        ]

    def get_milvus_id_value(self, instance: Model) -> Int64:
        high, mid, low = self.get_django_pk_values(instance)
        return high ^ mid ^ low

    def get_django_pk_values(self, instance: Model) -> List[Int64]:
        """Returns [django_pk_high, django_pk_mid, django_pk_low]"""
        if isinstance(instance.pk, int):
            pk = instance.pk  # int64
        elif isinstance(instance.pk, UUID):
            pk = instance.pk.int  # int128
        else:
            raise NotImplementedError()
        mask_high = 0b11
        mask_mid = mask_low = (1 << 63) - 1
        high = (pk >> 126) & mask_high
        mid = (pk >> 63) & mask_mid
        low = pk & mask_low
        assert high.bit_length() <= 2, f"{high.bit_length()} > 2"
        assert mid.bit_length() <= 63, f"{mid.bit_length()} > 63"
        assert low.bit_length() <= 63, f"{low.bit_length()} > 63"
        return [high, mid, low]

    def update_entry(self, instance: Model) -> None:
        self.check_schema(instance._meta.model)
        self.delete_entry(instance)
        self.insert_entry(instance)

    def bulk_update_entries(self, queryset: QuerySet) -> None:
        self.bulk_delete_entries(queryset)
        self.bulk_insert_entries(queryset)

    def check_schema(self, model: Type[Model]) -> None:
        current_fields = self.get_milvus_field_schemas(model)
        collection_fields: List[FieldSchema] = self.get_collection(model).schema.fields
        expected = [f.attname for f in current_fields]
        actual = [f.name for f in collection_fields]
        if expected != expected:
            raise ValueError(f"Schema mismatch: {expected=} {actual=}")

    def delete_entry(self, instance: Model) -> None:
        # pk = instance.pk
        # collection = self.get_collection(instance._meta.model)
        # collection.delete(pk)
        ...

    def bulk_delete_entries(self, queryset: QuerySet) -> None:
        ...

    def insert_entry(self, instance: Model) -> None:
        collection = self.get_collection(instance._meta.model)
        rows = [self.get_milvus_values(instance)]
        collection.insert(transpose(rows))

    def bulk_insert_entries(self, queryset: QuerySet) -> None:
        if queryset:
            collection = self.get_collection(queryset.model)
            rows = self.get_bulk_milvus_values(queryset)
            collection.insert(transpose(rows))

    def get_milvus_values(self, instance: Model) -> List[int]:
        """Returns a row of milvus values to insert."""
        fields = self.get_sorted_model_fields(instance._meta.model)
        values = [getattr(instance, f.attname) for f in fields]
        row = [
            self.get_milvus_id_value(instance),
            *self.get_django_pk_values(instance),
            *values,
        ]
        return row

    def get_bulk_milvus_values(self, queryset: QuerySet) -> List[List[Any]]:
        values: List[List[Any]] = []
        for q in queryset:
            values.append(self.get_milvus_values(q))
        return values

    def flush(self, model: Type[Model]) -> None:
        pymilvus.utility.get_connection().flush([self.get_collection_name(model)])


T = TypeVar("T")


def transpose(values: List[List[T]]) -> List[List[T]]:
    """The collection.insert() API takes columns instead of rows. Use this
    helper to transpose the input if needed.s"""
    return list(zip(*values))  # type:ignore
