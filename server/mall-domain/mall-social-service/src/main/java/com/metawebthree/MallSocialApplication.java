package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.mallsocial",
        "com.metawebthree.live",
        "com.metawebthree.groupbuying",
        "com.metawebthree.socialcommerce"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.live",
                "com.metawebthree.groupbuying",
                "com.metawebthree.socialcommerce"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
@EnableDubbo
@EnableScheduling
public class MallSocialApplication {

    public static void main(String[] args) {
        SpringApplication.run(MallSocialApplication.class, args);
    }

}