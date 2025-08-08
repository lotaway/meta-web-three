package com.metawebthree.common.cloud;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

@Slf4j
@Configuration
public class DefaultS3Config {

    @Value("${aws.region:未配置}")
    protected String region;

    @PostConstruct
    public void init() {
        log.info("DefaultS3Config region: " + region);
    }

    @Bean
    public S3Client s3Client() {
        return S3Client.builder()
                .region(Region.of(this.region))
                .build();
    }

}
