package com.metawebthree;

import com.metawebthree.user.infrastructure.config.DefaultAdminProperties;
import com.metawebthree.user.infrastructure.config.SeedDataProperties;
import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
@EnableDubbo
@EnableConfigurationProperties({DefaultAdminProperties.class, SeedDataProperties.class})
public class UserServiceApplication extends BaseApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}