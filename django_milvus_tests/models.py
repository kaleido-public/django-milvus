import random
from typing import List
from uuid import uuid4

from django.db.models import Model
from django.db.models.fields import UUIDField
from pymilvus.client.types import DataType

from django_milvus.fields import MilvusField


def random_vector(dim: int) -> List[int]:
    return [random.randrange(-100, 100) for _ in range(dim)]


def random_vector_2():
    return random_vector(2)


def random_vector_16():
    return random_vector(16)


class Product(Model):
    similarity = MilvusField(
        dim=2, dtype=DataType.FLOAT_VECTOR, default=random_vector_2
    )
    largefield = MilvusField(
        dim=16, dtype=DataType.FLOAT_VECTOR, default=random_vector_16
    )


class ProductUUID(Model):
    id = UUIDField(primary_key=True, default=uuid4)
    similarity = MilvusField(
        dim=2, dtype=DataType.FLOAT_VECTOR, default=random_vector_2
    )
