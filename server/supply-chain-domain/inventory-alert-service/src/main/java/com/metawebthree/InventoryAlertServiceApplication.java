package com.metawebthree;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "com.metawebthree")
@MapperScan("com.metawebthree.inventoryalert.infrastructure.persistence.mapper")
public class InventoryAlertServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(InventoryAlertServiceApplication.class, args);
    }
}