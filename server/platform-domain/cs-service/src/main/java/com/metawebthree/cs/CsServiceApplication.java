package com.metawebthree.cs;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = "com.metawebthree")
@EnableDiscoveryClient
@EnableDubbo
@MapperScan({"com.metawebthree.cs.infrastructure.persistence.mybatis", "com.metawebthree.common.audit"})
public class CsServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(CsServiceApplication.class, args);
    }
}
