"""Compute the class count of every registered classification task with SQL.

`num_classes` is declared per task, and it has to match the label encoding that
`ImputeEntityTaskMixin._init_target_mapping` builds: the sorted distinct
non-missing values of the target column of the whole entity table. That is a
`COUNT(DISTINCT ...)` on the remote database, so there is no need to download
anything.

All CTU databases live on one server, so a single connection with fully
qualified table names covers every task.

Usage:

    uv run python scripts/num-classes.py                  # every task
    uv run python scripts/num-classes.py --tasks cora-original genes-original
    uv run python scripts/num-classes.py --task-type binary
"""

import argparse
import re
from collections import defaultdict
from typing import Optional

import sqlalchemy as sa
from relbench.base import TaskType
from redelex.datasets import dataset_registry
from redelex.tasks import task_registry

import redelex  # noqa: F401  (registers the CTU datasets and tasks)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

TASK_TYPES = {
    "multiclass": TaskType.MULTICLASS_CLASSIFICATION,
    "binary": TaskType.BINARY_CLASSIFICATION,
}

# A few targets are not raw columns: the dataset's customize_db builds them. The
# query has to mirror that construction, so it is spelled out here per task.
DERIVED_TARGETS = {
    # VisualGenome.customize_db denormalizes OBJ_CLASSES into IMG_OBJ with a
    # right join, so the classes are the ones IMG_OBJ actually references.
    "visualgenome-original": """
        SELECT COUNT(*) AS total, COUNT(c.OBJ_CLASS) AS non_null,
               COUNT(DISTINCT c.OBJ_CLASS) AS distinct_collated,
               COUNT(DISTINCT BINARY c.OBJ_CLASS) AS distinct_binary
        FROM `VisualGenome`.`IMG_OBJ` o
        LEFT JOIN `VisualGenome`.`OBJ_CLASSES` c ON o.OBJ_CLASS_ID = c.OBJ_CLASS_ID
    """,
}


def quote(identifier: str) -> str:
    """Backtick-quote a MariaDB identifier, refusing anything unexpected."""
    if not IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Refusing to interpolate the identifier {identifier!r}")
    return f"`{identifier}`"


def collect_tasks(task_types: list[TaskType], only: Optional[set[str]] = None):
    """Group the registered tasks of the given types by their database."""
    by_database = defaultdict(list)

    for dataset_name in sorted(n for n in dataset_registry if n.startswith("ctu-")):
        dataset_cls, dataset_args, dataset_kwargs = dataset_registry[dataset_name]
        dataset = dataset_cls(*dataset_args, **{**dataset_kwargs, "cache_dir": None})

        for task_name, (task_cls, _, _) in sorted(task_registry.get(dataset_name, {}).items()):
            if task_cls.task_type not in task_types:
                continue
            if only is not None and task_name not in only and dataset_name not in only:
                continue
            by_database[dataset.database].append((dataset_name, task_name, task_cls))

    return by_database


def existing_columns(con: sa.Connection, database: str) -> dict[str, set[str]]:
    """Map table name -> column names, for one database, in a single query."""
    rows = con.execute(
        sa.text(
            "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = :db"
        ),
        {"db": database},
    ).fetchall()

    columns = defaultdict(set)
    for table_name, column_name in rows:
        columns[table_name].add(column_name)
    return columns


def count_classes(con: sa.Connection, database: str, table: str, column: str) -> dict:
    """Distinct non-NULL values of a column, as the task's label encoding sees it."""
    qualified = f"{quote(database)}.{quote(table)}"
    col = quote(column)

    # BINARY forces byte comparison. Without it MariaDB applies the column's
    # collation, which can fold case and trailing spaces and therefore count
    # fewer classes than pandas' factorize does.
    row = con.execute(
        sa.text(
            f"SELECT COUNT(*) AS total, COUNT({col}) AS non_null, "
            f"COUNT(DISTINCT {col}) AS distinct_collated, "
            f"COUNT(DISTINCT BINARY {col}) AS distinct_binary "
            f"FROM {qualified}"
        )
    ).one()

    return {
        "total": row.total,
        "non_null": row.non_null,
        "collated": row.distinct_collated,
        "binary": row.distinct_binary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-type",
        choices=[*TASK_TYPES, "all"],
        default="multiclass",
        help="Which task types to report (default: multiclass).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Restrict to these task or dataset names.",
    )
    args = parser.parse_args()

    task_types = (
        list(TASK_TYPES.values())
        if args.task_type == "all"
        else [TASK_TYPES[args.task_type]]
    )
    by_database = collect_tasks(task_types, set(args.tasks) if args.tasks else None)
    if not by_database:
        print("No matching tasks.")
        return

    # Every CTU database is on the same server, so one dataset's URL gets us a
    # connection for all of them; the database is named explicitly per query.
    any_dataset_cls, any_args, any_kwargs = dataset_registry["ctu-financial"]
    url = sa.engine.make_url(
        any_dataset_cls(*any_args, **{**any_kwargs, "cache_dir": None}).remote_url
    ).set(database="")

    engine = sa.create_engine(url)
    results = []
    problems = []

    with engine.connect() as con:
        for database, tasks in sorted(by_database.items()):
            columns = existing_columns(con, database)

            for dataset_name, task_name, task_cls in tasks:
                table, column = task_cls.entity_table, task_cls.target_col
                declared = getattr(task_cls, "num_classes", None)

                if task_name in DERIVED_TARGETS:
                    row = con.execute(sa.text(DERIVED_TARGETS[task_name])).one()
                    results.append(
                        {
                            "dataset": dataset_name,
                            "task": task_name,
                            "cls": task_cls.__name__,
                            "table": f"{database}.{table}",
                            "column": f"{column} (derived)",
                            "declared": declared,
                            "total": row.total,
                            "non_null": row.non_null,
                            "collated": row.distinct_collated,
                            "binary": row.distinct_binary,
                        }
                    )
                    continue

                if table not in columns:
                    problems.append(
                        f"{dataset_name}/{task_name}: entity table '{table}' is not in "
                        f"database '{database}' (built by customize_db?)"
                    )
                    continue
                if column not in columns[table]:
                    problems.append(
                        f"{dataset_name}/{task_name}: target column '{column}' is not in "
                        f"'{database}.{table}' (built by customize_db?)"
                    )
                    continue

                counts = count_classes(con, database, table, column)
                results.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "cls": task_cls.__name__,
                        "table": f"{database}.{table}",
                        "column": column,
                        "declared": declared,
                        **counts,
                    }
                )

    engine.dispose()

    width = max((len(f"{r['dataset']}/{r['task']}") for r in results), default=20)
    print(f"\n{'task'.ljust(width)}  classes  declared  rows  nulls  target")
    print("-" * (width + 46))
    for r in results:
        nulls = r["total"] - r["non_null"]
        declared = "-" if r["declared"] is None else str(r["declared"])
        print(
            f"{f'{r['dataset']}/{r['task']}'.ljust(width)}  "
            f"{r['binary']:>7}  {declared:>8}  {r['total']:>4}  {nulls:>5}  "
            f"{r['table']}.{r['column']}"
        )

    mismatched = [r for r in results if r["declared"] not in (None, r["binary"])]
    collation = [r for r in results if r["collated"] != r["binary"]]

    print("\n# Paste into the task definitions in redelex/tasks/ctu_tasks.py:")
    for r in results:
        print(f"class {r['cls']}:  num_classes = {r['binary']}")

    if collation:
        print(
            "\n! Collation folds values in these columns; the byte-wise count above is "
            "what pandas sees:"
        )
        for r in collation:
            print(
                f"  {r['dataset']}/{r['task']}: collated={r['collated']} "
                f"binary={r['binary']} ({r['table']}.{r['column']})"
            )

    if mismatched:
        print("\n! Already declared, but disagreeing with the data:")
        for r in mismatched:
            print(f"  {r['dataset']}/{r['task']}: declared={r['declared']} actual={r['binary']}")

    if problems:
        print("\n! Not resolvable with plain SQL:")
        for problem in problems:
            print(f"  {problem}")


if __name__ == "__main__":
    main()
