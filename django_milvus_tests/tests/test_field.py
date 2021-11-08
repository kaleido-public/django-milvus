from django.test import TestCase

from django_milvus.fields import MilvusField, rebuild_index, update_entry
from django_milvus_tests.models import Product


class TestMilvusField(TestCase):
    def test_milvusfield_save_empty(self):
        product = Product.objects.create(similarity=[])
        self.assertEqual([], product.similarity)

    def test_milvusfield_save_int(self):
        product = Product.objects.create(similarity=[12, 34])
        self.assertEqual([12, 34], product.similarity)

    def test_milvusfield_save_float(self):
        product = Product.objects.create(similarity=[12.34, 56.78])
        self.assertEqual([12.34, 56.78], product.similarity)

    def test_milvusfield_lookup(self):
        p1 = Product.objects.create(similarity=[0, 0])
        p2 = Product.objects.create(similarity=[12.34, 56.78])
        rebuild_index(Product)

        actual = Product.objects.filter(similarity__nearest_1=[-1, -1]).first()
        self.assertEqual(p1, actual)

        actual = Product.objects.filter(similarity__nearest_1=[99, 99]).first()
        self.assertEqual(p2, actual)
