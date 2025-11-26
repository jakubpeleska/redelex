import os
import re
import tempfile

import jinja2
import pydot

from .interface import DBInterface

TPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpl")


def visualize_db(
    interface: DBInterface, db_name: str, output_dir: str, hide_columns=False
) -> str:
    """
    Visualizes the database schema using Graphviz and Jinja2 templates.

    Args:
        interface (DBInterface): The database interface to extract schema from.
        db_name (str): The name of the database.
        output_dir (str): The directory to save the output SVG file.
        hide_columns (bool): Whether to hide column details in the visualization.
    Returns:
        str: The path to the generated SVG file.
    """

    dbschema = interface.get_schema()

    # Data structure to hold the schema, matching the template's 'tables' structure
    graph = {
        "name": db_name,
        "disablefields": hide_columns,
        "tables": [],
    }

    for table_name in dbschema.table_names:
        table_schema = dbschema.table_schemas[table_name]
        table = {
            "id": _name_to_id(table_name),
            "name": table_name,
            "fields": [],
            "relations": [],
        }

        for col, col_type in table_schema.type_dict.items():
            table["fields"].append({"name": col, "type": col_type, "blank": False})

        for fk in table_schema.fks:
            relation = {
                "target": _name_to_id(fk.ref_table),
                "name": "__".join(fk.src_columns),
                "arrows": "crow",
            }
            table["relations"].append(relation)

        graph["tables"].append(table)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template("sqlviz.tpl")

    dot_str = template.render(graph)

    os.makedirs(output_dir, exist_ok=True)

    out_filename = os.path.join(output_dir, f"{db_name}.svg")
    tmp_dot_file = os.path.join(tempfile.gettempdir(), f"{db_name}.dot")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=True) as tmp_dot:
        tmp_dot.write(dot_str)
        tmp_dot_file = tmp_dot.name
        pydot_graph = pydot.graph_from_dot_file(tmp_dot_file)[0]
        output_graphviz_svg = pydot_graph.write(out_filename, format="svg")

    if not output_graphviz_svg:
        raise RuntimeError("Graphviz failed to generate the SVG file.")

    return out_filename


def _name_to_id(name: str) -> str:
    return re.sub(r"[\s-]", "", name)


__all__ = ["visualize_db"]
