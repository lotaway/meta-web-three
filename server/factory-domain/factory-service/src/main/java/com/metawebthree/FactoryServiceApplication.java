package com.metawebthree;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Import;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.DatabaseConfig;
import com.metawebthree.common.config.SwaggerConfig;
import com.metawebthree.common.config.WebConfig;
import com.metawebthree.common.interceptor.SecurityInterceptor;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.config",
        "com.metawebthree.mes",
        "com.metawebthree.digitaltwin",
        "com.metawebthree.production",
        "com.metawebthree.common.event"
})
@Import({
        CommonSecurityConfig.class,
        DatabaseConfig.class,
        SwaggerConfig.class,
        WebConfig.class,
        SecurityInterceptor.class
})
@MapperScan(
        basePackages = {
                "com.metawebthree.mes",
                "com.metawebthree.digitaltwin",
                "com.metawebthree.production"
        },
        annotationClass = Mapper.class
)
@EnableDiscoveryClient
@EnableScheduling
public class FactoryServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(FactoryServiceApplication.class, args);
    }

}
