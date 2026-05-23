package com.metawebthree.digitaltwin.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("production_lines")
public class ProductionLineDO extends BaseDO {
    private Long id;
    private String lineCode;
    private String lineName;
    private String workshopId;
    private String status;
    private Integer capacity;
    private Integer currentOutput;
    private Double efficiency;
    private String productTypes;
}
