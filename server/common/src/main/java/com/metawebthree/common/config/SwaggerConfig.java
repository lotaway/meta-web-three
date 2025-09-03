package com.metawebthree.common.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SwaggerConfig {

        @Bean
        public OpenAPI customOpenAPI() {
                return new OpenAPI()
                                .info(new Info()
                                                .title("Meta Web Three - Service API")
                                                .version("1.0.0")
                                                .description(System.getProperty("spring.boot.project.name", "")
                                                                + "API documentation for the Service of Meta Web Three platform")
                                                .contact(new Contact()
                                                                .name("Meta Web Three Team")
                                                                .email("support@metawebthree.com")
                                                                .url("https://metawebthree.com"))
                                                .license(new License()
                                                                .name("MIT License")
                                                                .url("https://opensource.org/licenses/MIT")));
        }
}
