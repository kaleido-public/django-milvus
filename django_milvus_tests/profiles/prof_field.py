import cProfile
import random

from django.test import TestCase

from django_milvus.utils import rebuild_index
from django_milvus_tests.models import Product


def random_vector(dim: int):
    return [random.randrange(-100, 100) for _ in range(dim)]


class TestMilvusField(TestCase):
    def test_repeated_search(self):
        for i in range(10000):
            p1 = Product.objects.create(similarity=random_vector(dim=2))
        rebuild_index(Product)
        with cProfile.Profile() as pr:
            for i in range(100):
                product = Product.objects.filter(
                    similarity__nearest_50=random_vector(dim=2)
                ).first()
                print(product)
        pr.dump_stats("cprofile")
