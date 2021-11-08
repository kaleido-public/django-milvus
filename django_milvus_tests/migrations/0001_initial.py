# Generated by Django 3.2.9 on 2021-11-08 19:13

from django.db import migrations, models
import django_milvus.fields
import pymilvus.client.types


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('similarity', django_milvus.fields.MilvusField(dbname='default', dim=2, dtype=pymilvus.client.types.DataType['FLOAT_VECTOR'])),
            ],
        ),
    ]
