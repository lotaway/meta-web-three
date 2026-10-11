package com.metawebthree.erp;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = {"com.metawebthree.erp", "com.metawebthree.hrm", "com.metawebthree.crm", "com.metawebthree.project"})
@MapperScan(basePackages = {"com.metawebthree.erp", "com.metawebthree.hrm", "com.metawebthree.crm", "com.metawebthree.project"}, annotationClass = Mapper.class)
@EnableDiscoveryClient
public class ErpManagementApplication {
    public static void main(String[] args) {
        SpringApplication.run(ErpManagementApplication.class, args);
    }
}