from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type
from uuid import UUID

from django.db.models import Model, UUIDField
from django.db.models.lookups import Lookup

if TYPE_CHECKING:
    from django_milvus.connection import Connection
    from django_milvus.fields import MilvusField


def get_nearest_n(
    count: int, model: Type[Model], field: MilvusField, connection: Connection
) -> Type[Lookup]:
    class NearestN(Lookup):
        def get_prep_lookup(self) -> List[Any]:
            """rhs is a vector. look up near vectors in milvus and return
            their pks"""
            target_vector = self.rhs
            connection.connect()
            collection = connection.get_collection(model)
            collection.load()
            # Search supports searching multiple target vectors at the same.
            # However, for our purpose, we just need to search one. SearchResult
            # is iterable and is a 2d-array-like class, the first dimension is
            # the number of vectors to query (nq), the second dimension is the
            # number of limit(topk).
            result = collection.search(
                [
                    target_vector
                ],  # the length of this is the 1st dimension of the result, in our case it's 1.
                field.attname,
                param={
                    "metric_type": field.metric_type,
                    "params": {"nprobe": field.nprobe},
                },
                limit=count,  # 2nd dimension of the search result.
            )
            ids = list(result[0].ids)
            result2: List[Dict[str, Any]] = collection.query(
                expr=f"id in {ids}",
                output_fields=[
                    "django_pk_high",
                    "django_pk_mid",
                    "django_pk_low",
                ],
            )
            django_pks: List[Any] = []
            for r in result2:
                high = r["django_pk_high"]
                mid = r["django_pk_mid"]
                low = r["django_pk_low"]
                mask_high = 0b11
                mask_mid = mask_low = (1 << 63) - 1
                django_pk = (
                    ((high & mask_high) << 126)
                    + ((mid & mask_mid) << 63)
                    + (low & mask_low)
                )
                django_pks.append(django_pk)
            if isinstance(model._meta.pk, UUIDField):
                django_pks = [UUID(int=x) for x in django_pks]
            return django_pks

        def get_db_prep_lookup(self, value, connection):
            return "(" + ",".join(["%s"] * len(value)) + ")", value

        def as_sql(self, compiler, connection):
            rhs, rhs_params = self.process_rhs(compiler, connection)
            assert model._meta.pk is not None
            return f"{model._meta.pk.name} IN {rhs}", rhs_params

    return NearestN
