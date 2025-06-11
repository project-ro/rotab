import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotab.core.pipeline import Pipeline

here = Path(__file__).parent.resolve()


# paths are relative to the current directory
if __name__ == "__main__":
    pipeline = Pipeline.from_setting(
        template_dir="./examples/config/templates",
        param_dir="./examples/config/params",
        schema_dir="./examples/config/schemas",
        derive_func_path="./examples/custom_functions/derive_funcs.py",
        transform_func_path="./examples/custom_functions/transform_funcs.py",
    )
    pipeline.run(execute=True, dag=True)
