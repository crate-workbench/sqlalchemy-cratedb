from urllib.parse import urlencode

from sqlalchemy_cratedb.support.model import TableOptions


ALL_OPTIONS = {
    "if-exists": "replace",
    "partitioned-by": "time",
    "clustered-by": '"A"',
    "replicas": "0-2",
    "shards": "2",
    "durability": "async",
    "max-fields": "42",
    "column-policy": "dynamic",
    "refresh-interval": "500",
    "routing-shards": "5",
    "disable-read": "true",
    "disable-write": "true",
    "disable-metadata": "true",
    "read-only": "true",
    "read-only-allow-delete": "true",
    "soft-deletes-enable": "false",
    "soft-deletes-retention-period": "48h",
    "compression": "deflate",
}


def test_table_options_success():
    url_all = "crate://crate@localhost:4200/testdrive/demo?" + urlencode(ALL_OPTIONS)
    options = TableOptions.from_url(url_all)
    assert options == {
        'crate_"blocks.metadata"': "true",
        'crate_"blocks.read"': "true",
        'crate_"blocks.read_only"': "true",
        'crate_"blocks.read_only_allow_delete"': "true",
        'crate_"blocks.write"': "true",
        'crate_"mapping.total_fields.limit"': 42,
        'crate_"soft_deletes.enabled"': "false",
        'crate_"soft_deletes.retention_lease.period"': "'48h'",
        'crate_"translog.durability"': "'async'",
        "crate_clustered_by": '"A"',
        "crate_codec": "'best_compression'",
        "crate_column_policy": "'dynamic'",
        "crate_number_of_replicas": "'0-2'",
        "crate_number_of_routing_shards": 5,
        "crate_number_of_shards": 2,
        "crate_partitioned_by": "time",
        "crate_refresh_interval": 500,
    }
