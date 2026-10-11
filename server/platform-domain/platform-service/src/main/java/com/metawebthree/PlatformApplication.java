package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Import;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.DatabaseConfig;
import com.metawebthree.common.config.RedisCacheConfig;
import com.metawebthree.common.config.RestClientConfig;
import com.metawebthree.common.config.SwaggerConfig;
import com.metawebthree.common.config.WebConfig;
import com.metawebthree.common.cloud.DefaultS3Config;
import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.common.cloud.LocalStorageService;
import com.metawebthree.common.cloud.MinioConfig;
import com.metawebthree.common.cloud.MinioService;
import com.metawebthree.common.exception.GlobalExceptionHandler;
import com.metawebthree.common.interceptor.SecurityInterceptor;
import com.metawebthree.common.services.DistributedCacheService;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
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
        "com.metawebthree.tenant",
        "com.metawebthree.common.audit"
})
@Import({
        CommonSecurityConfig.class,
        DatabaseConfig.class,
        RestClientConfig.class,
        RedisCacheConfig.class,
        SwaggerConfig.class,
        WebConfig.class,
        SecurityInterceptor.class,
        MinioConfig.class,
        MinioService.class,
        DefaultS3Config.class,
        DefaultS3Service.class,
        LocalStorageService.class,
        MQProducer.class,
        DistributedCacheService.class,
        GlobalExceptionHandler.class
})
@MapperScan(
        basePackages = {
                "com.metawebthree.message",
                "com.metawebthree.media",
                "com.metawebthree.commission",
                "com.metawebthree.cs",
                "com.metawebthree.dataanalysis",
                "com.metawebthree.developerportal",
                "com.metawebthree.tenant",
                "com.metawebthree.common.audit"
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
