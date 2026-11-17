package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.media.config.UploadQuotaProperties;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.config",
        "com.metawebthree.message",
        "com.metawebthree.email",
        "com.metawebthree.job",
        "com.metawebthree.tgbot",
        "com.metawebthree.media",
        "com.metawebthree.commission",
        "com.metawebthree.cs",
        "com.metawebthree.dataanalysis",
        "com.metawebthree.developerportal",
        "com.metawebthree.tenant"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.message",
                "com.metawebthree.media",
                "com.metawebthree.commission",
                "com.metawebthree.cs",
                "com.metawebthree.dataanalysis",
                "com.metawebthree.developerportal",
                "com.metawebthree.tenant"
        },
        annotationClass = Mapper.class
)
@EnableConfigurationProperties(UploadQuotaProperties.class)
@EnableDiscoveryClient
@EnableDubbo
@EnableScheduling
public class PlatformApplication {

    public static void main(String[] args) {
        SpringApplication.run(PlatformApplication.class, args);
    }

}
