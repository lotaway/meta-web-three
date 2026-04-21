package com.metawebthree.promotion;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.quartz.QuartzAutoConfiguration;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication(exclude = QuartzAutoConfiguration.class)
@EnableDiscoveryClient
@EnableDubbo
public class PromotionServiceApplication extends BaseApplication {
    public static void main(String[] args) {
        SpringApplication.run(PromotionServiceApplication.class, args);
    }
}
