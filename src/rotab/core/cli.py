import argparse
from rotab.runtime.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run a ROTAB data processing pipeline.")

    parser.add_argument("--template-dir", type=str, required=True, help="Directory containing YAML templates")
    parser.add_argument("--param-dir", type=str, help="Directory containing parameter YAMLs", default=None)
    parser.add_argument("--schema-dir", type=str, required=True, help="Directory containing schema YAMLs")
    parser.add_argument("--derive-func-path", type=str, help="Path to custom derive function definitions", default=None)
    parser.add_argument(
        "--transform-func-path", type=str, help="Path to custom transform function definitions", default=None
    )
    parser.add_argument(
        "--source-dir", type=str, default=".generated", help="Output directory for generated code and data"
    )
    parser.add_argument("--execute", action="store_true", help="Execute the generated code")
    parser.add_argument("--dag", action="store_true", help="Generate a DAG (Mermaid format)")
    args = parser.parse_args()

    print("=== ROTAB Pipeline Configuration ===")
    print(f"Template directory      : {args.template_dir}")
    print(f"Parameter directory     : {args.param_dir or '(none)'}")
    print(f"Schema directory        : {args.schema_dir}")
    print(f"Derive function path    : {args.derive_func_path or '(none)'}")
    print(f"Transform function path : {args.transform_func_path or '(none)'}")
    print(f"Output directory        : {args.source_dir}")
    print(f"Execute                 : {'Yes' if args.execute else 'No'}")
    print(f"Generate DAG            : {'Yes' if args.dag else 'No'}\n")

    pipeline = Pipeline.from_setting(
        template_dir=args.template_dir,
        source_dir=args.source_dir,
        param_dir=args.param_dir,
        schema_dir=args.schema_dir,
        derive_func_path=args.derive_func_path,
        transform_func_path=args.transform_func_path,
    )

    if args.dag:
        print("→ Generating Mermaid DAG...")
        pipeline.generate_dag(source_dir=args.source_dir)

    print("→ Generating code" + (" and executing..." if args.execute else " only..."))
    pipeline.run(
        execute=args.execute,
        dag=args.dag,
    )

    print("\nPipeline completed successfully.")
    if not args.execute:
        print(f"To run the generated code manually:\n   python {args.source_dir}/main.py")
