package com.metawebthree.finance;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication(scanBasePackages = {"com.metawebthree.finance", "com.metawebthree.invoice", "com.metawebthree.reporting", "com.metawebthree.settlement"})
@MapperScan(basePackages = {"com.metawebthree.finance", "com.metawebthree.invoice", "com.metawebthree.reporting", "com.metawebthree.settlement"}, annotationClass = Mapper.class)
@EnableDiscoveryClient
@EnableScheduling
public class FinanceServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(FinanceServiceApplication.class, args);
    }
}