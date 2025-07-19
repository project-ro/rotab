import os
import pandas as pd
import csv
import questionary
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.allow_unicode = True


def infer_schema_from_csv(path, schema_name):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 最初の行だけ読む（超高速）

    df = pd.read_csv(path, usecols=header, nrows=100)

    schema = CommentedMap()
    schema["name"] = schema_name
    schema.yaml_set_comment_before_after_key("name", before="Name of the schema")
    schema["description"] = "Inferred schema from sample CSV"
    schema.yaml_set_comment_before_after_key("description", before="What this dataset represents")

    columns = CommentedSeq()
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith("int"):
            schema_type = "int"
        elif dtype.startswith("float"):
            schema_type = "float"
        elif dtype.startswith("bool"):
            schema_type = "bool"
        else:
            schema_type = "str"
        col_entry = CommentedMap()
        col_entry["name"] = col
        col_entry["dtype"] = schema_type
        col_entry["description"] = f"Inferred as {schema_type}"
        columns.append(col_entry)

    schema["columns"] = columns
    schema.yaml_set_comment_before_after_key("columns", before="List of fields with types and meaning")
    return schema


def build_output_schema(output_schema_name):
    schema = CommentedMap()
    schema["name"] = output_schema_name
    schema.yaml_set_comment_before_after_key("name", before="Schema name for output")

    schema["description"] = "Schema for processed output file"
    schema.yaml_set_comment_before_after_key("description", before="Brief description of output content")

    columns = CommentedSeq()

    for name, dtype, desc in [
        ("id", "str", "Renamed from user_id"),
        ("age", "int", "Passed through"),
        ("age_group", "int", "Derived column: age // 10 * 10"),
    ]:
        col = CommentedMap()
        col["name"] = name
        col["dtype"] = dtype
        col["description"] = desc
        columns.append(col)

    schema["columns"] = columns
    schema.yaml_set_comment_before_after_key("columns", before="Output columns with explanation")
    return schema


def build_template(input_schema_name, output_schema_name):
    tpl = CommentedMap()

    tpl["name"] = "main_template"
    tpl.yaml_set_comment_before_after_key("name", before="Name of this Rotab template")

    tpl["depends"] = []
    tpl.yaml_set_comment_before_after_key("depends", before="Templates to run before this one")

    proc = CommentedMap()
    proc["name"] = "simple_process"
    proc.yaml_set_comment_before_after_key("name", before="Process name")
    proc["description"] = "Basic filtering and transformation"

    io = CommentedMap()

    io["inputs"] = CommentedSeq(
        [{"name": input_schema_name, "io_type": "csv", "path": "../../source/input.csv", "schema": input_schema_name}]
    )
    io.yaml_set_comment_before_after_key("inputs", before="Input data sources and their schema")

    io["outputs"] = CommentedSeq(
        [
            {
                "name": output_schema_name,
                "io_type": "csv",
                "path": "../../output/result.csv",
                "schema": output_schema_name,
            }
        ]
    )
    io.yaml_set_comment_before_after_key("outputs", before="Output destination and expected schema")

    proc["io"] = io

    steps = CommentedSeq()

    step1 = CommentedMap()
    step1["name"] = "basic_filter"
    step1.yaml_set_comment_before_after_key("name", before="Step name")
    step1["with"] = input_schema_name
    step1.yaml_set_comment_before_after_key("with", before="Which input variable to use")
    step1["mutate"] = [
        {"filter": "age > ${params.min_age}"},
        {"derive": "age_group = age // 10 * 10"},
        {"select": ["user_id", "age", "age_group"]},
    ]
    step1.yaml_set_comment_before_after_key("mutate", before="Operations to apply to the dataset")
    step1["as"] = "filtered"
    step1.yaml_set_comment_before_after_key("as", before="Alias name for the result of this step")

    step2 = CommentedMap()
    step2["name"] = "finalize"
    step2["with"] = "filtered"
    step2["transform"] = "rename(col='user_id', to='id')"
    step2["as"] = output_schema_name

    steps.append(step1)
    steps.append(step2)

    proc["steps"] = steps

    tpl["processes"] = [proc]
    tpl.yaml_set_comment_before_after_key("processes", before="List of all processes in this template")

    return tpl


def build_params():
    params = CommentedMap()
    values = CommentedMap()
    values["min_age"] = 20
    params["params"] = values
    params.yaml_set_comment_before_after_key("params", before="All parameters referenced in the template")
    return params


def initialize_project():
    project_name = questionary.text("project name:").ask()
    if not project_name:
        print("Project name cannot be empty.")
        return

    input_dir = questionary.path("input data directory").ask()
    backend = questionary.select("choose backend", choices=["polars", "pandas"]).ask()

    root = os.path.abspath(project_name)
    config_dir = os.path.join(root, "config")
    schema_dir = os.path.join(config_dir, "schemas")
    param_dir = os.path.join(config_dir, "params")
    template_dir = os.path.join(config_dir, "templates")
    custom_func_dir = os.path.join(root, "custom_functions")

    os.makedirs(schema_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(custom_func_dir, exist_ok=True)

    first_schema_name = None

    if input_dir and os.path.isdir(input_dir):
        for file in os.listdir(input_dir):
            if file.endswith(".csv"):
                csv_found = True
                full_path = os.path.join(input_dir, file)
                schema_name = os.path.splitext(file)[0]
                if first_schema_name is None:
                    first_schema_name = schema_name

                input_schema = infer_schema_from_csv(full_path, schema_name)
                with open(os.path.join(schema_dir, f"{schema_name}.yaml"), "w", encoding="utf-8") as f:
                    yaml.dump(input_schema, f)

    template = build_template(input_schema_name=first_schema_name, output_schema_name="output_data")
    with open(os.path.join(template_dir, "template.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(template, f)

    params = build_params()
    with open(os.path.join(param_dir, "params.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(params, f)

    for kind in ["derive", "transform"]:
        fname = f"{kind}_funcs_{backend}.py"
        fpath = os.path.join(custom_func_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(f"# {kind.capitalize()} functions for {backend}\n")

    print(f"\nProject '{project_name}' initialized (backend: {backend})\n → {root}")
