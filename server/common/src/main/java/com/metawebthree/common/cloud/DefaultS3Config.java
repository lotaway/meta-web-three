package com.metawebthree.common.cloud;

import java.nio.file.Path;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.auth.credentials.AwsCredentials;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.profiles.ProfileFile;
import software.amazon.awssdk.profiles.ProfileFile.Type;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

@Slf4j
@Configuration
public class DefaultS3Config {

    @Value("${aws.region:未配置}")
    protected String region;

    @Value("${aws.s3.buckets.name:metaserver}")
    protected String name;

    @PostConstruct
    public void init() {
        log.info("DefaultS3Config region: " + region + ", name: " + name);
    }

    @Bean
    public S3Client s3Client() {
        try {
            Path dockerCredentialsPath = Path.of("/.aws/credentials");
            if (java.nio.file.Files.exists(dockerCredentialsPath)) {
                log.info("Using AWS credentials from Docker environment: {}", dockerCredentialsPath);
                ProfileFile profile = ProfileFile.builder().content(dockerCredentialsPath)
                        .type(Type.CONFIGURATION)
                        .build();
                ProfileCredentialsProvider credentialsProvider = ProfileCredentialsProvider.builder()
                        .profileFile(profile)
                        .profileName("default").build();

                return S3Client.builder()
                        .credentialsProvider(credentialsProvider)
                        .region(Region.of(this.region))
                        .build();
            }
            Path localCredentialsPath = Path.of(".aws/credentials");
            if (java.nio.file.Files.exists(localCredentialsPath)) {
                log.info("Using AWS credentials from local environment: {}", localCredentialsPath);
                ProfileFile profile = ProfileFile.builder().content(localCredentialsPath)
                        .type(Type.CONFIGURATION)
                        .build();
                ProfileCredentialsProvider credentialsProvider = ProfileCredentialsProvider.builder()
                        .profileFile(profile)
                        .profileName("default").build();

                return S3Client.builder()
                        .credentialsProvider(credentialsProvider)
                        .region(Region.of(this.region))
                        .build();
            }

            // 如果两个路径都不存在，使用默认配置
            log.warn(
                    "AWS credentials file not found at Docker path {} or local path {}, using default configuration without credentials",
                    dockerCredentialsPath, localCredentialsPath);
            return S3Client.builder()
                    .region(Region.of(this.region))
                    .build();
        } catch (Exception e) {
            log.warn("Error creating S3 client, using default configuration. Error: {}", e.getMessage());
            return S3Client.builder()
                    .region(Region.of(this.region))
                    .build();
        }
    }

    public String getName() {
        return name;
    }

}
