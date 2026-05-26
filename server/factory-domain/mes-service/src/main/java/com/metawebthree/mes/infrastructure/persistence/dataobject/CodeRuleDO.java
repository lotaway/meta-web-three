package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.time.LocalDateTime;

/**
 * 编码规则 DO
 */
@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("mes_code_rule")
public class CodeRuleDO {
    
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String businessType;
    private String ruleExpression;
    private Long startValue;
    private Long currentValue;
    private Integer step;
    private Integer paddingLength;
    private String elements; // JSON 序列化的规则元素列表
    private String status; // ACTIVE, INACTIVE
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;
}