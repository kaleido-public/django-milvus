from django.test import TestCase

from django_milvus.utils import rebuild_index
from django_milvus_tests.models import ProductUUID


class TestUUIDPrimaryField(TestCase):
    def test_build_db(self):
        rebuild_index(ProductUUID)

    def test_milvusfield_lookup_uuid(self):
        prod = ProductUUID.objects.create(similarity=[12.34, 56.78])
        rebuild_index(ProductUUID)

        actual = ProductUUID.objects.filter(similarity__nearest_1=[99, 99]).first()
        self.assertEqual(prod, actual)
