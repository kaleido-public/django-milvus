# type:ignore

import pymilvus.client.types
from django.db import migrations, models

import django_milvus.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Product",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "similarity",
                    django_milvus.fields.MilvusField(
                        dbname="default",
                        dim=2,
                        dtype=pymilvus.client.types.DataType["FLOAT_VECTOR"],
                        index_type="IVF_FLAT",
                        metric_type="L2",
                        nlist=1024,
                        nprobe=32,
                    ),
                ),
                (
                    "largefield",
                    django_milvus.fields.MilvusField(
                        dbname="default",
                        dim=16,
                        dtype=pymilvus.client.types.DataType["FLOAT_VECTOR"],
                        index_type="IVF_FLAT",
                        metric_type="L2",
                        nlist=1024,
                        nprobe=32,
                    ),
                ),
            ],
        ),
    ]
