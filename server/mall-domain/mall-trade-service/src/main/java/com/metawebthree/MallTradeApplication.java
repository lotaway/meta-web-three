package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Import;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.services.QuartzManager;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.malltrade",
        "com.metawebthree.config",
        "com.metawebthree.order",
        "com.metawebthree.payment",
        "com.metawebthree.cart",
        "com.metawebthree.promotion",
        "com.metawebthree.aftersale"
})
@Import(QuartzManager.class)
@MapperScan(
        basePackages = {
                "com.metawebthree.order",
                "com.metawebthree.payment",
                "com.metawebthree.cart",
                "com.metawebthree.promotion",
                "com.metawebthree.aftersale"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
@EnableDubbo
@EnableScheduling
@EnableAsync
public class MallTradeApplication {

    public static void main(String[] args) {
        SpringApplication.run(MallTradeApplication.class, args);
    }

}