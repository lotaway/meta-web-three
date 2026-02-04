package com.metawebthree.config;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

import com.metawebthree.common.config.MybatisPlusDefaultConfig;

@Configuration
@MapperScan("com.metawebthree")
public class MybatisPlusConfig extends MybatisPlusDefaultConfig {
}
