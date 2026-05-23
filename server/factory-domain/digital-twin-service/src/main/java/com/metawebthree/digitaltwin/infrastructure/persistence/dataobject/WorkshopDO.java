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
@TableName("workshops")
public class WorkshopDO extends BaseDO {
    private Long id;
    private String workshopCode;
    private String workshopName;
    private String description;
    private String status;
    private Integer area;
    private String location;
    private Double centerX;
    private Double centerY;
    private Double width;
    private Double length;
}
