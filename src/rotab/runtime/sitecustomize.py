# sitecustomize.py
import os, tempfile

TARGET = "/opt/ml/tmp"
os.makedirs(TARGET, exist_ok=True)

os.environ["TMPDIR"] = TARGET
os.environ["TEMP"] = TARGET
os.environ["TMP"] = TARGET
os.environ["XDG_CACHE_HOME"] = TARGET
os.environ["JUPYTER_RUNTIME_DIR"] = TARGET
os.environ["ARROW_TMPDIR"] = TARGET
os.environ["POLARS_TEMP_DIR"] = TARGET
os.environ["JOBLIB_TEMP_FOLDER"] = TARGET

tempfile.tempdir = TARGET
