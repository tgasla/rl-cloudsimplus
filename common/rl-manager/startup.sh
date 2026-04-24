#!/bin/bash
# Generate protobuf files if they don't exist or are outdated

PROTO_SOURCE="/papers/main/cloudsimplus-gateway/src/main/proto/cloudsimplus.proto"
PROTO_OUT_DIR="/mgr/gym_cloudsimplus/gym_cloudsimplus"
PB2_FILE="$PROTO_OUT_DIR/cloudsimplus_pb2.py"
PB2_GRPC_FILE="$PROTO_OUT_DIR/cloudsimplus_pb2_grpc.py"

# Check if we need to regenerate
if [ ! -f "$PB2_FILE" ] || [ ! -f "$PB2_GRPC_FILE" ] || \
   [ "$PROTO_SOURCE" -nt "$PB2_FILE" ]; then
    echo "Generating gRPC protobuf files..."
    pip install grpcio grpcio-tools --quiet 2>/dev/null
    cd "$PROTO_OUT_DIR"
    python -m grpc_tools.protoc \
        -I/papers/main/cloudsimplus-gateway/src/main/proto \
        --python_out=. \
        --grpc_python_out=. \
        "$PROTO_SOURCE"
    echo "gRPC protobuf files generated."
fi

export PYTHONPATH="/mgr/gym_cloudsimplus/gym_cloudsimplus:$PYTHONPATH"
exec python entrypoint.py
