import os

# Tests run on CPU regardless of the machine; the backend is chosen before JAX initialises.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
