package com.metawebthree.aftersale.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.Contact;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * OpenAPI Configuration for After-Sale Service
 * Provides API documentation via /v3/api-docs endpoint
 */
@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI afterSaleServiceOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("After-Sale Service API")
                        .description("API documentation for After-Sale Management Service")
                        .version("1.0.0")
                        .contact(new Contact()
                                .name("Meta Web Three Team")
                                .email("support@metawebthree.com")));
    }
}