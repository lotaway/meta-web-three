package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.config",
        "com.metawebthree.forecasting",
        "com.metawebthree.routeoptimizer",
        "com.metawebthree.ai"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.forecasting",
                "com.metawebthree.routeoptimizer"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
@EnableDubbo
public class AiServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiServiceApplication.class, args);
    }

}
