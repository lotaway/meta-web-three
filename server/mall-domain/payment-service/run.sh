#!/bin/bash

# Payment Service 启动脚本

echo "Starting Payment Service..."

# 检查Java版本
java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
echo "Java version: $java_version"

# 检查Maven是否安装
if ! command -v mvn &> /dev/null; then
    echo "Maven is not installed. Please install Maven first."
    exit 1
fi

# 清理并编译
echo "Cleaning and compiling..."
mvn clean compile

# 运行测试
echo "Running tests..."
mvn test

# 打包
echo "Building package..."
mvn package -DskipTests

# 运行应用
echo "Starting application..."
java -jar target/payment-service-*.jar 