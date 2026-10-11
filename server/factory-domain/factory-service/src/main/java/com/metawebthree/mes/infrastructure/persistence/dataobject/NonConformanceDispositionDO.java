package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_qc_non_conformance")
public class NonConformanceDispositionDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String dispositionCode;
    private String dispositionName;
    private String type;
    private String stepsJson;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}