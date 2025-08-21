package com.metawebthree.entity;

import com.metawebthree.enums.DecisionEnum;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class Rule {
    private String code;
    private String expr;
    private DecisionEnum action;
    private int priority;
}
