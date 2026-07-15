package com.metawebthree.dom.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class SourcingRuleDTO {
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
