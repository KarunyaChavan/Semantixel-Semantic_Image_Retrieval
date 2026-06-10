#!/usr/bin/env python3
"""Generate gRPC Python stubs from proto/semantixel_inference.proto.

Usage:
    python scripts/generate_proto.py

Output:
    semantixel/semantixel_inference_pb2.py
    semantixel/semantixel_inference_pb2_grpc.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO_DIR = os.path.join(PROJECT_ROOT, "proto")
OUT_DIR = os.path.join(PROJECT_ROOT, "semantixel")
PROTO_FILE = os.path.join(PROTO_DIR, "semantixel_inference.proto")


def main():
    from grpc_tools import protoc

    args = [
        "protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        PROTO_FILE,
    ]

    result = protoc.main(args)
    if result != 0:
        print(f"Error: protoc returned exit code {result}", file=sys.stderr)
        sys.exit(result)

    print(f"Stubs generated in {OUT_DIR}")

    grpc_stub = os.path.join(OUT_DIR, "semantixel_inference_pb2_grpc.py")
    with open(grpc_stub, "r") as f:
        content = f.read()

    content = content.replace(
        "import semantixel_inference_pb2 as semantixel__inference__pb2",
        "from semantixel import semantixel_inference_pb2 as semantixel__inference__pb2",
    )

    with open(grpc_stub, "w") as f:
        f.write(content)

    print(f"Patched import in {grpc_stub}")
    print("Done.")


if __name__ == "__main__":
    main()
