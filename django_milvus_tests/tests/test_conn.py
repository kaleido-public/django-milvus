from django.test import TestCase

from django_milvus.fields import rebuild_index, update_entry
from django_milvus_tests.models import Product


class TestMilvusConnection(TestCase):
    def test_connection(self):
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        product = Product.objects.create(similarity=[12, 34])
        rebuild_index(Product)

    def test_update_entry(self):
        for i in range(10):
            product = Product.objects.create(similarity=[12, 34])
            update_entry(product)
