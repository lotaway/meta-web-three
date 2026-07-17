package com.metawebthree.dom.application.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import java.time.LocalDateTime;

@Data
public class SourcingRuleDTO {
    private Long id;
    @NotBlank
    private String ruleName;
    @NotBlank
    private String ruleType;
    @NotNull
    private Integer priority;
    private String warehouseIds;
    private String region;
    private Boolean enabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
