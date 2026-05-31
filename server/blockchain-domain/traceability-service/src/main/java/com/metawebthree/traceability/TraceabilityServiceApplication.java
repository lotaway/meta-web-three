package com.metawebthree.traceability;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.metawebthree.traceability.infrastructure.persistence.mapper")
public class TraceabilityServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(TraceabilityServiceApplication.class, args);
    }
}