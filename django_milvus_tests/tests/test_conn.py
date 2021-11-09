from django.test import TestCase

from django_milvus.utils import rebuild_index, update_entry
from django_milvus_tests.models import Product
import random

def random_vector(dim: int):
    return [random.randrange(-100, 100) for _ in range(dim)]


class TestMilvusConnection(TestCase):
    def test_connection(self):
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        rebuild_index(Product)

    def test_update_entry(self):
        rebuild_index(Product)
        for i in range(10):
            product = Product.objects.create(similarity=[12, 34])
            update_entry(product)
