package com.metawebthree.common.VO;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "page-config")
public class PageConfigVO {
    private Integer pageSize;
}
