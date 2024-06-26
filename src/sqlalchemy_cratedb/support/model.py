import dataclasses
import typing as t
from collections import OrderedDict
from urllib.parse import urlparse, parse_qs, parse_qsl

from sqlalchemy.util import asbool


@dataclasses.dataclass
class TableOptionSpec:
    name: str
    type: t.Optional[str] = None
    default: t.Optional[t.Any] = None
    choices: t.Optional[list] = None
    translate: t.Optional[t.Dict[str, str]] = None
    unit: t.Optional[str] = None
    description: t.Optional[str] = None
    docs: t.Optional[str] = None


class TableOptions(dict):
    """
    Manage a dictionary of SQLAlchemy dialect options in `<dialect>_<option> = <value>` format.

    References:
    - https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#with
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-translog.html
    """

    # Map shorthand URL parameters to CrateDB's first-level special SQLAlchemy table options.
    URL_PARAM_SA_OPTIONS_MAP: t.Dict[str, TableOptionSpec] = {
        # Cluster options.
        "clustered-by": TableOptionSpec(name="crate_clustered_by"),
        "partitioned-by": TableOptionSpec(name="crate_partitioned_by"),
        "shards": TableOptionSpec(name="crate_number_of_shards", type="int", default=4),
        # Table options.
        "async-flush-interval": TableOptionSpec(
            name='crate_"translog.sync_interval"',
            type="int",
            unit="ms",
            default=5000,
            description="Transaction log flushing interval when using `durability=async`.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#translog-sync-interval",
        ),
        "async-flush-size": TableOptionSpec(
            name='crate_"translog.flush_threshold_size"',
            type="str",
            unit="bytes",
            default="512Mb",
            description="Transaction log flushing threshold size when using `durability=async`.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#translog-flush-threshold-size",
        ),
        "column-policy": TableOptionSpec(
            name="crate_column_policy",
            type="str",
            choices=["strict", "dynamic"],
            default="strict",
            description="The column policy of the table. "
            "`strict` will reject any column which is not defined in the schema. "
            "`dynamic` means that new columns can be added on demand.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#column-policy",
        ),
        "compression": TableOptionSpec(
            name="crate_codec",
            type="str",
            choices=["lz4", "deflate"],
            default="lz4",
            description="By default, data is stored using LZ4 compression. This can be changed "
            "to `best_compression` which uses DEFLATE for a higher compression ratio, "
            "at the expense of slower column value lookups..",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#codec",
            translate={
                "lz4": "default",
                "deflate": "best_compression",
            },
        ),
        "disable-read": TableOptionSpec(
            name='crate_"blocks.read"',
            type="bool",
            default=False,
            description="Set to `true` to disable all read operations for a table.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#blocks-read",
        ),
        "disable-metadata": TableOptionSpec(
            name='crate_"blocks.metadata"',
            type="bool",
            default=False,
            description="Set to `true` to disable modifications on table settings.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#blocks-metadata",
        ),
        "disable-write": TableOptionSpec(
            name='crate_"blocks.write"',
            type="bool",
            default=False,
            description="Set to `true` to disable all write operations for a table.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#blocks-write",
        ),
        "durability": TableOptionSpec(
            name='crate_"translog.durability"',
            type="str",
            choices=["request", "async"],
            default="request",
            description="If set to `request`, the transaction log will be flushed after every write operation. "
            "If set to `async`, the transaction log gets flushed to disk periodically in the background. ",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#translog-durability",
        ),
        "max-fields": TableOptionSpec(
            name='crate_"mapping.total_fields.limit"',
            type="int",
            default=1000,
            description="Maximum number of columns that is allowed for a table, "
            "including both the user facing mapping (columns) and internal fields.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#mapping-total-fields-limit",
        ),
        "max-ngram-diff": TableOptionSpec(
            name="crate_max_ngram_diff",
            type="int",
            default=1,
            description="Maximum difference between `max_ngram` and `min_ngram` when using the `NGramTokenizer` or the `NGramTokenFilter`.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#max-ngram-diff",
        ),
        "max-merge-threads": TableOptionSpec(
            name='crate_"merge.scheduler.max_thread_count"',
            type="int",
            description="Maximum number of threads on a single shard that may be merging at once. "
            "The default will be auto-configured based on the number of CPU cores, "
            "optimized for a good solid-state-disk (SSD). On spinning drives, decrease "
            "this to 1.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#merge-scheduler-max-thread-count",
        ),
        "max-shingle-diff": TableOptionSpec(
            name="crate_max_shingle_diff",
            type="int",
            default=3,
            description="Maximum difference between `min_shingle_size` and `"
            "max_shingle_size` when using the `ShingleTokenFilter`.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#max-shingle-diff",
        ),
        "min-shards-write": TableOptionSpec(
            name='crate_"write.wait_for_active_shards"',
            type="str",
            default="1",
            description="Number of shard copies that need to be active for write operations to proceed.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#write-wait-for-active-shards",
        ),
        "read-only": TableOptionSpec(
            name='crate_"blocks.read_only"',
            type="bool",
            default=False,
            description="Table is read only if value set to `true`. "
            "Allows writes and table settings changes if set to `false`.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#blocks-read-only",
        ),
        "read-only-allow-delete": TableOptionSpec(
            name='crate_"blocks.read_only_allow_delete"',
            type="bool",
            default=False,
            description="Allows to have a read only table that additionally can be deleted.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#blocks-read-only-allow-delete",
        ),
        "refresh-interval": TableOptionSpec(
            name="crate_refresh_interval",
            type="int",
            unit="ms",
            default=1000,
            description="Interval for table-level automatic background refresh.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#refresh-interval",
        ),
        "replicas": TableOptionSpec(
            name="crate_number_of_replicas",
            type="str",
            default="0-1",
            description="The number or range of replicas each shard of a table should have for normal operation",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#number-of-replicas",
        ),
        "routing-shards": TableOptionSpec(
            name="crate_number_of_routing_shards",
            type="int",
            # TODO: Which default value does this have?
            description="The hashing space that is used internally to distribute documents across shards.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#number-of-routing-shards",
        ),
        "routing-max-shards-per-node": TableOptionSpec(
            name='crate_"routing.allocation.total_shards_per_node"',
            type="int",
            default=-1,
            description="Control the total number of shards (replicas and primaries) "
            "allowed to be allocated on a single node. The default is unbounded.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#routing-allocation-total-shards-per-node",
        ),
        "routing-policy": TableOptionSpec(
            name='crate_"routing.allocation.enable"',
            type="str",
            choices=["all", "primaries", "new_primaries", "none"],
            default="all",
            description="Control shard allocation for a specific table.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#routing-allocation-enable",
        ),
        "routing-max-retries": TableOptionSpec(
            name='crate_"allocation.max_retries"',
            type="int",
            default=5,
            description="Attempts to allocate a shard before giving up and leaving the shard unallocated.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#allocation-max-retries",
        ),
        "routing-include": TableOptionSpec(
            name='crate_"routing.allocation.include.{attribute}"',
            type="str",
            description="Assign the table to a node whose `{attribute} has at least one of the comma-separated values.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#routing-allocation-include-attribute",
        ),
        "routing-require": TableOptionSpec(
            name='crate_"routing.allocation.require.{attribute}"',
            type="str",
            description="Assign the table to a node whose `{attribute} has all of the comma-separated values.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#routing-allocation-require-attribute",
        ),
        "routing-exclude": TableOptionSpec(
            name='crate_"routing.allocation.exclude.{attribute}"',
            type="str",
            description="Assign the table to a node whose `{attribute} has none of the comma-separated values.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#routing-allocation-exclude-attribute",
        ),
        "shard-unassigned-reallocate-delay": TableOptionSpec(
            name='crate_"unassigned.node_left.delayed_timeout"',
            type="str",
            default="1m",
            description="Delay the allocation of replica shards which have become unassigned because a node has left.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#unassigned-node-left-delayed-timeout",
        ),
        "soft-deletes-enable": TableOptionSpec(
            name='crate_"soft_deletes.enabled"',
            type="bool",
            default=True,
            description="Whether soft deletes are enabled or disabled.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#soft-deletes-enabled",
        ),
        "soft-deletes-retention-period": TableOptionSpec(
            name='crate_"soft_deletes.retention_lease.period"',
            type="str",
            default="12h",
            description="The maximum period for which a retention lease is retained before it is considered expired.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#soft-deletes-retention-lease-period",
        ),
        "store-type": TableOptionSpec(
            name='crate_"store.type"',
            type="str",
            choices=["fs", "niofs", "mmapfs", "hybridfs"],
            default="fs",
            description="Control how data is stored and accessed on disk. It's not possible to update this setting after table creation.",
            docs="https://cratedb.com/docs/crate/reference/en/latest/sql/statements/create-table.html#store-type",
        ),
    }

    # List of option name prefixes for second-level table options.
    URL_PARAM_SA_OPTIONS_PREFIXES = [
        "allocation.",
        "blocks.",
        "codec",
        "mapping.",
        "merge.",
        "routing.",
        "soft_deletes.",
        "store.",
        "translog.",
        "unassigned.",
        "write.",
    ]

    @classmethod
    def from_url(cls, url: t.Any):
        """
        Factory for creating a DialectOptions object from a URL object.
        """

        # Support for plain strings.
        if isinstance(url, str):
            url = urlparse(url)
            query_params = dict(parse_qsl(url.query))

        # Support for boltons.urlutils, as used by CrateDB Toolkit.
        elif hasattr(url, "query_params"):
            query_params = url.query_params
        else:
            raise TypeError(f"URL data type unsupported: {type(url)}")

        return cls.from_query_params(query_params=query_params)

    @classmethod
    def from_query_params(cls, query_params: t.Dict[str, str]):
        """
        Convert from shorthand URL parameter notation into a dictionary of SQLAlchemy dialect options.

        The output item format is `<dialect>_<option> = <value>`.
        """
        options = OrderedDict()

        # Process options having explicit mapping rules.
        for query_name, spec in cls.URL_PARAM_SA_OPTIONS_MAP.items():
            if query_name in query_params:
                options[spec.name] = cls.convert_value(query_params[query_name], spec)

        # Process options without explicit rules.
        for query_name_prefix in cls.URL_PARAM_SA_OPTIONS_PREFIXES:
            for param, value in query_params.items():
                key = f'crate_"{param}"'
                if param.startswith(query_name_prefix) and key not in options:
                    options[key] = value

        return cls(**options)

    @staticmethod
    def convert_value(value: t.Any, spec: TableOptionSpec) -> t.Union[str, int, None]:
        """
        Convert option value into the right format, based on its specification.
        """
        print("value:", value)
        if type_ := spec.type:
            if type_ == "str":
                value = value.strip("'")
            elif type_ == "int":
                value = value.strip("'")
                value = int(value)
            elif type_ == "bool":
                value = value.strip("'")
                value = asbool(value)
        if spec.choices:
            if value not in spec.choices:
                raise ValueError(f"Value {value} not permitted. Allowed choices: {spec.choices}")
        if spec.translate:
            value = spec.translate.get(value, value)
        if type_ == "str":
            value = f"'{value}'"
        elif type_ == "bool":
            value = str(value).lower()
        return value
