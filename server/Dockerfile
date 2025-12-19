FROM maven:3.9.11-eclipse-temurin-17 AS builder
WORKDIR /build
COPY pom.xml .
COPY */pom.xml ./
RUN mvn -B -e -C dependency:go-offline

COPY . .
RUN mvn -B package -DskipTests