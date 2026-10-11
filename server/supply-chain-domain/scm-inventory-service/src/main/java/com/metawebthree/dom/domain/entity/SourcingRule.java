package com.metawebthree.dom.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class SourcingRule {
    private Long id;
    private String ruleName;
    private String ruleType;
    private Integer priority;
    private String warehouseIds;
    private String region;
    private Boolean enabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
