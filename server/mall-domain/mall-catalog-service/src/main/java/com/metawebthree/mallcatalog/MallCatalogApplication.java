package com.metawebthree.mallcatalog;

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
import com.metawebthree.common.config.SwaggerConfig;
import com.metawebthree.recommendation.infrastructure.aishopping.config.AiShoppingProperties;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.mallcatalog",
        "com.metawebthree.config",
        "com.metawebthree.product",
        "com.metawebthree.image",
        "com.metawebthree.supplier",
        "com.metawebthree.recommendation",
        "com.metawebthree.review",
        "com.metawebthree.common.event",
        "com.metawebthree.common.elasticsearch",
        "com.metawebthree.common.utils.RocketMQ"
})
@MapperScan(
        basePackages = {
                "com.metawebthree.product",
                "com.metawebthree.image",
                "com.metawebthree.supplier",
                "com.metawebthree.recommendation",
                "com.metawebthree.review"
        },
        annotationClass = Mapper.class
)
@EnableConfigurationProperties({
        RecommendationAlgorithmProperties.class,
        AiShoppingProperties.class
})
@Import({
        CommonSecurityConfig.class,
        DatabaseConfig.class,
        SwaggerConfig.class
})
@EnableDiscoveryClient
@EnableScheduling
public class MallCatalogApplication {

    public static void main(String[] args) {
        SpringApplication.run(MallCatalogApplication.class, args);
    }
}