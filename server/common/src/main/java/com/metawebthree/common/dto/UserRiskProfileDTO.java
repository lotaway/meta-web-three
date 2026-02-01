package com.metawebthree.common.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserRiskProfileDTO implements Serializable {
    private Long userId;
    private Integer age;
    private Float externalDebtRatio;
    private Float gpsStability;
    private Integer deviceSharedDegree;
    private String deviceRiskTag; // Stored as String for enum mapping
}
