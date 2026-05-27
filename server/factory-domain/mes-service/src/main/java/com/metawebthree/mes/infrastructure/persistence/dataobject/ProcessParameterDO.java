package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 工艺参数 DO
 */
@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("mes_process_parameter")
public class ProcessParameterDO {
    
    private Long id;
    private String paramCode;
    private String paramName;
    private Long routeId;
    private String routeCode;
    private Integer stepNo;
    private String stepCode;
    private String paramType;
    private String dataType;
    private String unit;
    private java.math.BigDecimal standardValue;
    private java.math.BigDecimal upperLimit;
    private java.math.BigDecimal lowerLimit;
    private String collectionMethod;
    private String deviceAddress;
    private Boolean isRequired;
    private String validationRule;
    private java.math.BigDecimal alarmThreshold;
    private Integer displayOrder;
    private String paramGroup;
    private String remark;
    private String status;
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;
}