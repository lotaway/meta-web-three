package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.RestClientConfig;
import com.metawebthree.common.config.SwaggerConfig;
import com.metawebthree.common.config.WebConfig;
import com.metawebthree.common.exception.GlobalExceptionHandler;
import com.metawebthree.common.interceptor.SecurityInterceptor;

/**
 * Security and web infrastructure that is on by default for every servlet
 * service. No {@code enabled} switch: services only override the beans they
 * need through normal Spring conditions.
 */
@AutoConfiguration
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
@Import({
        CommonSecurityConfig.class,
        WebConfig.class,
        SecurityInterceptor.class,
        RestClientConfig.class,
        SwaggerConfig.class,
        GlobalExceptionHandler.class
})
public class CommonWebAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(PasswordEncoder.class)
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
