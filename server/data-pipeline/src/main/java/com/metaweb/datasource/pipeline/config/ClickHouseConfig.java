package com.metaweb.datasource.pipeline.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;

import javax.sql.DataSource;

@Slf4j
@Configuration
public class ClickHouseConfig {

    private static final int QUERY_TIMEOUT_SECONDS = 300;

    @Value("${clickhouse.url:jdbc:clickhouse://localhost:8123/meta_web_analytics}")
    private String clickhouseUrl;

    @Value("${clickhouse.username:default}")
    private String clickhouseUsername;

    @Value("${clickhouse.password:}")
    private String clickhousePassword;

    @Bean("clickHouseDataSource")
    public DataSource clickHouseDataSource() {
        DriverManagerDataSource dataSource = new DriverManagerDataSource();
        dataSource.setDriverClassName("com.clickhouse.jdbc.ClickHouseDriver");
        dataSource.setUrl(clickhouseUrl);
        dataSource.setUsername(clickhouseUsername);
        dataSource.setPassword(clickhousePassword);
        log.info("ClickHouse DataSource initialized: {}", clickhouseUrl);
        return dataSource;
    }

    @Bean("clickHouseJdbcTemplate")
    public JdbcTemplate clickHouseJdbcTemplate(DataSource clickHouseDataSource) {
        JdbcTemplate template = new JdbcTemplate(clickHouseDataSource);
        template.setQueryTimeout(QUERY_TIMEOUT_SECONDS);
        log.info("ClickHouse JdbcTemplate initialized");
        return template;
    }
}
