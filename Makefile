PROTO_DIR = protos
JAVA_OUT = server/common/src/main/java
PY_OUT = risk-scorer
RUST_DIR = order-match
RUST_OUT = $(RUST_DIR)/src/generated/rpc

PROTO_FILES := $(wildcard $(PROTO_DIR)/*.proto)

all: gen-java gen-python gen-rust

install:
	brew install protobuf
	python3 -m pip install --user grpcio grpcio-tools kazoo
	mkdir -p $(PROTO_DIR)

gen-java:
	@echo "Generating Java code..."
	@mkdir -p $(JAVA_OUT)
	protoc -I=$(PROTO_DIR) \
		--java_out=$(JAVA_OUT) \
		--grpc-java_out=$(JAVA_OUT) \
		$(PROTO_FILES)

gen-python:
	@echo "Generating Python code..."
	@mkdir -p $(PY_OUT)
	python -m grpc_tools.protoc -I$(PROTO_DIR) \
		--python_out=$(PY_OUT) \
		--grpc_python_out=$(PY_OUT) \
		--pyi_out=$(PY_OUT) \
		$(PROTO_FILES)
	# protoc --proto_path=$(PROTO_DIR) --python_out=$(PY_OUT) --pyi_out=$(PY_OUT) $(PROTO_FILES)

gen-rust:
	@echo "Generating Rust code..."
	@mkdir -p $(RUST_OUT)
	@cd $(RUST_DIR) && RUST_BACKTRACE=1 cargo build

clean:
	rm -rf $(JAVA_OUT)/com/metawebthree/common/generated/rpc/*.java
	rm -rf $(PY_OUT)/*.py
	rm -rf $(RUST_DIR)/src/generated/rpc/*.rs
