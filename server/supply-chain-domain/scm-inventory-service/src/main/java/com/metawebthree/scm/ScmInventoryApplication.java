package com.metawebthree.scm;

import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.FullyQualifiedAnnotationBeanNameGenerator;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.scm",
        "com.metawebthree.config",
        "com.metawebthree.inventory",
        "com.metawebthree.warehouse",
        "com.metawebthree.supplier",
        "com.metawebthree.dom",
        "com.metawebthree.inventoryalert"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.inventory",
                "com.metawebthree.warehouse",
                "com.metawebthree.supplier",
                "com.metawebthree.dom",
                "com.metawebthree.inventoryalert"
        },
        annotationClass = Mapper.class,
        nameGenerator = FullyQualifiedAnnotationBeanNameGenerator.class
)
@EnableFeignClients(basePackages = "com.metawebthree.supplier.infrastructure.rpc")
@EnableDiscoveryClient
public class ScmInventoryApplication {
    public static void main(String[] args) {
        SpringApplication.run(ScmInventoryApplication.class, args);
    }
}
