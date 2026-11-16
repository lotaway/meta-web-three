package com.metawebthree;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

// scanBasePackages lists only this service's own packages. Shared security
// infrastructure (InternalTokenUtil, InternalTokenFilter, RequirePermissionAspect)
// is supplied by the common module's auto-configuration; adding a common
// subpackage here would make one service behave differently from the rest.
@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.config",
        "com.metawebthree.mes",
        "com.metawebthree.digitaltwin",
        "com.metawebthree.production"
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
