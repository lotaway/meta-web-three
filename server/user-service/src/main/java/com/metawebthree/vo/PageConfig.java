package com.metawebthree.vo;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "page-config")
public class PageConfig {
    private Integer pageSize;
}
