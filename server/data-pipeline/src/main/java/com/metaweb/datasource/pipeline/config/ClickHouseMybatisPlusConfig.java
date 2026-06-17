package com.metaweb.datasource.pipeline.config;

import com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionTemplate;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;

@Configuration
@MapperScan(
    basePackages = "com.metaweb.datasource.pipeline.repository.mapper",
    sqlSessionFactoryRef = "clickHouseSqlSessionFactory"
)
public class ClickHouseMybatisPlusConfig {

    @Bean("clickHouseSqlSessionFactory")
    public SqlSessionFactory clickHouseSqlSessionFactory(
            @Qualifier("clickHouseDataSource") DataSource dataSource) throws Exception {
        MybatisSqlSessionFactoryBean factoryBean = new MybatisSqlSessionFactoryBean();
        factoryBean.setDataSource(dataSource);
        return factoryBean.getObject();
    }

    @Bean("clickHouseSqlSessionTemplate")
    public SqlSessionTemplate clickHouseSqlSessionTemplate(
            @Qualifier("clickHouseSqlSessionFactory") SqlSessionFactory sqlSessionFactory) {
        return new SqlSessionTemplate(sqlSessionFactory);
    }
}
