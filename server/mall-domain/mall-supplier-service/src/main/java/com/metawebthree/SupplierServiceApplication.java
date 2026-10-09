package com.metawebthree;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "com.metawebthree")
@MapperScan({"com.metawebthree.supplier.infrastructure.persistence.mapper", "com.metawebthree.common.audit"})
public class SupplierServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(SupplierServiceApplication.class, args);
    }
}