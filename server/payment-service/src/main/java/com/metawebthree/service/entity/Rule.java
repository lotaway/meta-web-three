package com.metawebthree.service.entity;

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
    private String action;
    private int priority;
}
