import argparse
from rotab.core.pipeline import Pipeline
from rotab.utils.logger import get_logger, configure_logger
from rotab.init.initialize import initialize_project


def main():
    parser = argparse.ArgumentParser(description="ROTAB: Run or initialize a structured data pipeline.")
    parser.add_argument("--init", action="store_true", help="Initialize a new ROTAB project (interactive)")

    parser.add_argument("--template-dir", type=str, help="YAML template directory")
    parser.add_argument("--param-dir", type=str, default=None, help="YAML parameter directory (optional)")
    parser.add_argument("--schema-dir", type=str, help="Schema YAML directory")
    parser.add_argument("--derive-func-path", type=str, default=None, help="Custom derive function path")
    parser.add_argument("--transform-func-path", type=str, default=None, help="Custom transform function path")
    parser.add_argument("--source-dir", type=str, default=".generated", help="Output directory")
    parser.add_argument("--backend", type=str, choices=["pandas", "polars"], default="pandas", help="Backend to use")
    parser.add_argument("--processes", type=str, nargs="+", default=None, help="Process names to include")
    parser.add_argument("--execute", action="store_true", help="Execute the generated pipeline")
    parser.add_argument("--dag", action="store_true", help="Generate DAG output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    configure_logger(level="DEBUG" if args.debug else "INFO")
    logger = get_logger()
    logger.info("ROTAB CLI started.")

    if args.init:
        # disallowed = [
        #     "--template-dir",
        #     "--param-dir",
        #     "--schema-dir",
        #     "--derive-func-path",
        #     "--transform-func-path",
        #     "--source-dir",
        #     "--backend",
        #     "--processes",
        #     "--execute",
        #     "--dag",
        # ]
        # for flag in disallowed:
        #     if getattr(args, flag.lstrip("--").replace("-", "_")) not in (None, False):
        #         parser.error(f"--init cannot be used with {flag}")

        initialize_project()
        logger.info("Project initialized successfully.")
        return

    if not args.template_dir or not args.schema_dir:
        parser.error("Must provide --template-dir and --schema-dir unless using --init")

    print("=== ROTAB Pipeline Configuration ===")
    print(f"Template directory      : {args.template_dir}")
    print(f"Parameter directory     : {args.param_dir or '(none)'}")
    print(f"Schema directory        : {args.schema_dir}")
    print(f"Derive function path    : {args.derive_func_path or '(none)'}")
    print(f"Transform function path : {args.transform_func_path or '(none)'}")
    print(f"Output directory        : {args.source_dir}")
    print(f"Backend                 : {args.backend}")
    print(f"Execute                 : {'Yes' if args.execute else 'No'}")
    print(f"Generate DAG            : {'Yes' if args.dag else 'No'}")
    print(f"Processes               : {args.processes or '(all)'}")

    pipeline = Pipeline.from_setting(
        template_dir=args.template_dir,
        source_dir=args.source_dir,
        param_dir=args.param_dir,
        schema_dir=args.schema_dir,
        derive_func_path=args.derive_func_path,
        transform_func_path=args.transform_func_path,
        backend=args.backend,
    )

    pipeline.run(execute=args.execute, dag=args.dag, selected_processes=args.processes)

    if args.execute:
        logger.info("Pipeline run completed successfully.")
    else:
        logger.info("Code generation completed successfully.")
        print(f"\nTo run the generated code manually:\n  python {args.source_dir}/main.py")
