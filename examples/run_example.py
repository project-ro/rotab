import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotab.core.pipeline import Pipeline

here = Path(__file__).parent.resolve()


# paths are relative to the current directory
if __name__ == "__main__":
    pipeline = Pipeline.from_template_dir(
        dirpath="./examples/config/templates",
        param_path="./examples/config/params/params.yaml",
        derive_func_paths=["./custom_functions/derive_funcs.py"],
        transform_func_paths=["./custom_functions/transform_funcs.py"],
    )
    pipeline.run(script_path="./scripts/generated_user_flow.py", execute=True, dag=True)
