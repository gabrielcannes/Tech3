import os
import sys
import importlib.util

print("Python:", sys.executable)

print("\nMPI-related environment variables:")

for key, value in os.environ.items():
    if "MPI" in key or "DYLD" in key:
        print(f"{key}: {value}")

print("\nmpi4py installed:")

spec = importlib.util.find_spec("mpi4py")
print(spec.origin if spec else "Not installed")