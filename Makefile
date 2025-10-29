PROTO_DIR = protos
JAVA_DIR = server/common
JAVA_OUT = $(JAVA_DIR)/src/main/java
PY_OUT = risk-scorer
RUST_DIR = order-match
RUST_OUT = $(RUST_DIR)/src/generated/rpc

PROTO_FILES := $(wildcard $(PROTO_DIR)/*.proto)
PROTO_INCLUDE := $(wildcard $(PROTO_DIR)/**/*.proto)

all: gen-java-dubbo gen-python gen-rust

install:
	brew install protobuf
	python3 -m pip install --user grpcio grpcio-tools kazoo
	mkdir -p $(PROTO_DIR)

gen-java-grpc:
	@echo "Generating Java GRPC code..."
	@mkdir -p $(JAVA_OUT)
	protoc -I=$(PROTO_DIR) \
		--java_out=$(JAVA_OUT) \
		--grpc-java_out=$(JAVA_OUT) \
		$(PROTO_FILES)

gen-java-dubbo:
	@echo "Generating Java Dubbo code..."
	@cd $(JAVA_DIR) && mvn -q -DskipTests protobuf:compile protobuf:compile-custom

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
	@cd $(RUST_DIR) && RUST_BACKTRACE=1 PROTOC_INCLUDE="$(PROTO_INCLUDE)" cargo build

clean:
	rm -rf $(JAVA_OUT)/com/metawebthree/common/generated/rpc/*.java
	rm -rf $(PY_OUT)/*_pb2.py $(PY_OUT)/*_pb2.pyi $(PY_OUT)/*_pb2_grpc.py
	rm -rf $(RUST_DIR)/src/generated/rpc/*.rs
