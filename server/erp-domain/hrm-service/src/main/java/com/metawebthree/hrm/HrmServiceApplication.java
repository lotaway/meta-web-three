package com.metawebthree.hrm;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.metawebthree.hrm.domain.repository")
public class HrmServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(HrmServiceApplication.class, args);
    }
}