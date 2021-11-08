from pymilvus.client.types import DataType
from django_milvus.fields import MilvusField
from django.db.models import Model


class Product(Model):
    similarity = MilvusField(dim=2, dtype=DataType.FLOAT_VECTOR)
