package com.metawebthree.common.config;

import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
public class CommonSecurityConfig {

    @Bean
    @Order(0)
    public SecurityFilterChain commonSecurityFilterChain(HttpSecurity http) throws Exception {
        http.securityMatcher("/v3/api-docs/**", "/swagger-ui/**", "/actuator/**")
            .authorizeHttpRequests(auth -> auth.anyRequest().permitAll())
            .httpBasic(httpBasic -> httpBasic.disable())
            .csrf(csrf -> csrf.disable());
        return http.build();
    }
}
