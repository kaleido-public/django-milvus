from __future__ import annotations

from typing import TYPE_CHECKING, Type

from django.db.models.base import Model
from django.db.models.lookups import Lookup

if TYPE_CHECKING:
    from django_milvus.connection import Connection
    from django_milvus.fields import MilvusField


def get_nearest_n(
    count: int, model: Type[Model], field: MilvusField, connection: Connection
) -> Type[Lookup]:
    class NearestN(Lookup):
        def get_prep_lookup(self):
            """rhs is a vector. look up near vectors in milvus and return
            their pks"""
            target_vector = self.rhs
            connection.connect()
            collection = connection.get_collection(model)
            collection.load()
            result = collection.search(
                [target_vector],
                field.attname,
                param={
                    "metric_type": field.metric_type,
                    "params": {"nprobe": field.nprobe},
                },
                limit=count,
            )
            ids = list(result[0].ids)  # type: ignore
            return ids

        def get_db_prep_lookup(self, value, connection):
            return "(" + ",".join(["%s"] * len(value)) + ")", value

        def as_sql(self, compiler, connection):
            rhs, rhs_params = self.process_rhs(compiler, connection)
            return f"{model._meta.pk.name} IN {rhs}", rhs_params  # type: ignore

    return NearestN
