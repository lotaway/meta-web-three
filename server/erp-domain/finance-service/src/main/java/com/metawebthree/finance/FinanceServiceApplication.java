package com.metawebthree.finance;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = {"com.metawebthree.finance", "com.metawebthree.invoice"})
@MapperScan(basePackages = {"com.metawebthree.finance", "com.metawebthree.invoice"}, annotationClass = Mapper.class)
@EnableDiscoveryClient
public class FinanceServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(FinanceServiceApplication.class, args);
    }
}