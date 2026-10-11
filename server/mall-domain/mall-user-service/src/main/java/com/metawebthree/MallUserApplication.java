package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.apache.ibatis.annotations.Mapper;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Import;

import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.VO.PageConfigVO;
import com.metawebthree.common.config.CommonSecurityConfig;
import com.metawebthree.common.config.DatabaseConfig;
import com.metawebthree.common.config.PasswordConfig;
import com.metawebthree.common.config.SwaggerConfig;
import com.metawebthree.common.config.WebConfig;
import com.metawebthree.common.interceptor.SecurityInterceptor;
import com.metawebthree.common.utils.JwtUtil;
import com.metawebthree.common.utils.UserJwtUtil;
import com.metawebthree.user.infrastructure.config.DefaultAdminProperties;
import com.metawebthree.user.infrastructure.config.SeedDataProperties;

@SpringBootApplication(scanBasePackages = {
        "com.metawebthree.malluser",
        "com.metawebthree.user",
        "com.metawebthree.author",
        "com.metawebthree.utils",
        "com.metawebthree.action",
        "com.metawebthree.riskcontrol"
})
@Import({
        CommonSecurityConfig.class,
        DatabaseConfig.class,
        PasswordConfig.class,
        SwaggerConfig.class,
        WebConfig.class,
        SecurityInterceptor.class,
        TokenBlacklistService.class,
        PageConfigVO.class,
        JwtUtil.class,
        UserJwtUtil.class
})
@MapperScan(
        basePackages = {
                "com.metawebthree.user",
                "com.metawebthree.author",
                "com.metawebthree.action",
                "com.metawebthree.riskcontrol"
        },
        annotationClass = Mapper.class
)
@EnableConfigurationProperties({
        DefaultAdminProperties.class,
        SeedDataProperties.class
})
@EnableDiscoveryClient
@EnableDubbo
public class MallUserApplication {

    public static void main(String[] args) {
        SpringApplication.run(MallUserApplication.class, args);
    }

}
