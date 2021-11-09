from django.db.models import Model
from pymilvus.client.types import DataType

from django_milvus.fields import MilvusField
import random


def random_vector(dim: int):
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
