package com.metawebthree;

import com.metawebthree.media.config.UploadQuotaProperties;
import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
@EnableDubbo
@EnableConfigurationProperties(UploadQuotaProperties.class)
public class MediaServiceApplication extends BaseApplication {

    public static void main(String[] args) {
        SpringApplication.run(MediaServiceApplication.class, args);
    }
}
