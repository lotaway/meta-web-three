package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

import com.metawebthree.user.infrastructure.config.DefaultAdminProperties;
import com.metawebthree.user.infrastructure.config.SeedDataProperties;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.malluser",
        "com.metawebthree.user",
        "com.metawebthree.author",
        "com.metawebthree.utils",
        "com.metawebthree.action",
        "com.metawebthree.riskcontrol"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.user",
                "com.metawebthree.author",
                "com.metawebthree.action",
                "com.metawebthree.riskcontrol"
        },
        annotationClass = Mapper.class
)
@EnableConfigurationProperties({
        DefaultAdminProperties.class,
        SeedDataProperties.class
})
@EnableDiscoveryClient
@EnableDubbo
public class MallUserApplication {

    public static void main(String[] args) {
        SpringApplication.run(MallUserApplication.class, args);
    }

}
