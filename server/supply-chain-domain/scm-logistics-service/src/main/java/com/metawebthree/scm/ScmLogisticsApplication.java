package com.metawebthree.scm;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.scm",
        "com.metawebthree.config",
        "com.metawebthree.logistics",
        "com.metawebthree.procurement",
        "com.metawebthree.rma"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.logistics",
                "com.metawebthree.procurement",
                "com.metawebthree.rma"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
public class ScmLogisticsApplication {
    public static void main(String[] args) {
        SpringApplication.run(ScmLogisticsApplication.class, args);
    }
}