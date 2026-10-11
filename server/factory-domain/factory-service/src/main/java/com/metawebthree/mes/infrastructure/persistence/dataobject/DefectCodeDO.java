package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_qc_defect_code")
public class DefectCodeDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String defectCode;
    private String defectName;
    private String category;
    private String severity;
    private Boolean isCritical;
    private String description;
    private String dispositionGuide;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}