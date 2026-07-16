# Makefile for meta-web-three
# Updated for new directory structure

PROTO_DIR = protos
JAVA_DIR = server/common
PYTHON_DIR = server/ai-domain/risk-scorer
JAVA_OUT = $(JAVA_DIR)/src/main/java

# Find all proto files recursively
PROTO_FILES := $(wildcard $(PROTO_DIR)/*.proto) $(wildcard $(PROTO_DIR)/*/*.proto) $(wildcard $(PROTO_DIR)/*/*/*.proto)

# Proto file categories
PROTO_MALL := $(wildcard $(PROTO_DIR)/mall/*.proto)
PROTO_SUPPLY_CHAIN := $(wildcard $(PROTO_DIR)/supply-chain/*.proto)
PROTO_AI := $(wildcard $(PROTO_DIR)/ai/*.proto)
PROTO_PLATFORM := $(wildcard $(PROTO_DIR)/platform/*.proto)
PROTO_ERP := $(wildcard $(PROTO_DIR)/erp/*.proto)
PROTO_FACTORY := $(wildcard $(PROTO_DIR)/factory/*.proto)
PROTO_SHARED := $(wildcard $(PROTO_DIR)/shared/*.proto)

all: gen-java-dubbo gen-python

install:
	@echo "Installing dependencies..."
	@which protoc || brew install protobuf
	python3 -m pip install --user grpcio grpcio-tools kazoo 2>/dev/null || echo "Python deps may already exist"
	@mkdir -p $(PROTO_DIR)

# Generate Java Dubbo code (uses Maven plugin in server/common)
gen-java-dubbo:
	@echo "Generating Java Dubbo code..."
	@cd $(JAVA_DIR) && mvn -q -DskipTests protobuf:generate

# Generate Python code for risk-scorer
gen-python:
	@echo "Generating Python code for AI domain..."
	@mkdir -p $(PYTHON_DIR)
	@if [ -n "$(PROTO_FILES)" ]; then \
		python -m grpc_tools.protoc -I$(PROTO_DIR) \
			--python_out=$(PYTHON_DIR) \
			--grpc_python_out=$(PYTHON_DIR) \
			--pyi_out=$(PYTHON_DIR) \
			$(PROTO_FILES); \
	fi

# Generate for specific domain
gen-erp:
	@echo "Generating code for ERP domain..."
	@mkdir -p $(JAVA_OUT)
	@if [ -n "$(PROTO_ERP)" ]; then \
		protoc -I=$(PROTO_DIR) \
			--java_out=$(JAVA_OUT) \
			--grpc-java_out=$(JAVA_OUT) \
			$(PROTO_ERP); \
	fi

gen-factory:
	@echo "Generating code for Factory domain..."
	@mkdir -p $(JAVA_OUT)
	@if [ -n "$(PROTO_FACTORY)" ]; then \
		protoc -I=$(PROTO_DIR) \
			--java_out=$(JAVA_OUT) \
			--grpc-java_out=$(JAVA_OUT) \
			$(PROTO_FACTORY); \
	fi

gen-mall:
	@echo "Generating code for Mall domain..."
	@mkdir -p $(JAVA_OUT)
	@if [ -n "$(PROTO_MALL)" ]; then \
		protoc -I=$(PROTO_DIR) \
			--java_out=$(JAVA_OUT) \
			--grpc-java_out=$(JAVA_OUT) \
			$(PROTO_MALL); \
	fi

gen-supply-chain:
	@echo "Generating code for Supply Chain domain..."
	@mkdir -p $(JAVA_OUT)
	@if [ -n "$(PROTO_SUPPLY_CHAIN)" ]; then \
		protoc -I=$(PROTO_DIR) \
			--java_out=$(JAVA_OUT) \
			--grpc-java_out=$(JAVA_OUT) \
			$(PROTO_SUPPLY_CHAIN); \
	fi

gen-ai:
	@echo "Generating code for AI domain..."
	@mkdir -p $(PYTHON_DIR)
	@if [ -n "$(PROTO_AI)" ]; then \
		python -m grpc_tools.protoc -I=$(PROTO_DIR) \
			--python_out=$(PYTHON_DIR) \
			--grpc_python_out=$(PYTHON_DIR) \
			--pyi_out=$(PYTHON_DIR) \
			$(PROTO_AI); \
	fi

# List available proto files
list-protos:
	@echo "Available proto files:"
	@echo "  Mall: $(PROTO_MALL)"
	@echo "  Supply Chain: $(PROTO_SUPPLY_CHAIN)"
	@echo "  AI: $(PROTO_AI)"
	@echo "  ERP: $(PROTO_ERP)"
	@echo "  Factory: $(PROTO_FACTORY)"
	@echo "  Platform: $(PROTO_PLATFORM)"
	@echo "  Shared: $(PROTO_SHARED)"

clean:
	@echo "Cleaning generated files..."
	@rm -rf $(JAVA_OUT)/com/metawebthree/common/generated/rpc/*.java
	@rm -rf $(PYTHON_DIR)/*_pb2.py $(PYTHON_DIR)/*_pb2.pyi $(PYTHON_DIR)/*_pb2_grpc.py
	@cd $(JAVA_DIR) && mvn -q clean 2>/dev/null || true

.PHONY: all install gen-java-dubbo gen-python gen-erp gen-factory gen-mall gen-supply-chain gen-ai list-protos clean