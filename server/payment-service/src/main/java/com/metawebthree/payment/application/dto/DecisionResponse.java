package com.metawebthree.payment.application.dto;

import java.util.List;

import com.metawebthree.enums.DecisionEnum;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class DecisionResponse {
    private DecisionEnum decision;
    private int score;
    private List<String> reasons;
}
