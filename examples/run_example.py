import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from rotab.core.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline.from_template_file("examples/config/example.yaml")
    pipeline.run()
