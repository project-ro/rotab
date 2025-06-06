import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotab.core.pipeline import Pipeline

here = Path(__file__).parent.resolve()

if __name__ == "__main__":
    pipeline = Pipeline.from_template_dir(
        dirpath="./config",
        define_func_paths=["../custom_functions/define_funcs.py"],
        transform_func_paths=["../custom_functions/transform_funcs.py"],
    )
    pipeline.run(script_path="./scripts/generated_user_flow.py", execute=True, dag=True)
