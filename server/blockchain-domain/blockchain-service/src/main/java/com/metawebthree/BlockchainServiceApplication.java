package com.metawebthree;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.config",
        "com.metawebthree.wallet",
        "com.metawebthree.traceability"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.wallet",
                "com.metawebthree.traceability"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
public class BlockchainServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(BlockchainServiceApplication.class, args);
    }

}
