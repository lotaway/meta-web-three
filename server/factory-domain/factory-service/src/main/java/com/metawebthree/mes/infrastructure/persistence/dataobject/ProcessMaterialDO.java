package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_material")
public class ProcessMaterialDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long processBomId;
    private String materialCode;
    private String materialName;
    private String materialSpec;
    private String unitCode;
    private String unitName;
    private Double quantity;
    private Double scrapRate;
    private Integer sequence;
    private String materialType;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}