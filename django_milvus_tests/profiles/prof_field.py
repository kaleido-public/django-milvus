import cProfile
import random
from typing import List

from django.test import TestCase

from django_milvus.utils import rebuild_index
from django_milvus_tests.models import Product


def random_vector(dim: int) -> List[int]:
    return [random.randrange(-100, 100) for _ in range(dim)]


class TestMilvusField(TestCase):
    def test_repeated_search(self):
        print("create items")
        Product.objects.bulk_create(
            [Product(largefield=random_vector(dim=16)) for _ in range(10 ** 6)]
        )
        print("rebuild_index")
        rebuild_index(Product)
        with cProfile.Profile() as pr:
            for i in range(100):
                product = Product.objects.filter(
                    largefield__nearest_50=random_vector(dim=16)
                ).first()
                print(product)
        pr.dump_stats("cprofile")
