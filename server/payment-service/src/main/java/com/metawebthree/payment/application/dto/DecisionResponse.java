package com.metawebthree.payment.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;

import com.metawebthree.payment.enums.DecisionEnum;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Schema(description = "风控决策响应")
public class DecisionResponse {
    @Schema(description = "决策结果")
    private DecisionEnum decision;
    @Schema(description = "风险分数")
    private int score;
    @Schema(description = "原因列表")
    private List<String> reasons;
}
