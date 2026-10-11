package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_checklist_record")
public class EquipmentChecklistRecordDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String recordCode;
    private Long equipmentId;
    private String equipmentCode;
    private Long templateId;
    private String templateCode;
    private LocalDateTime checkPlanTime;
    private LocalDateTime checkActualTime;
    private String checkerId;
    private String checkerName;
    private String status;
    private Integer totalItems;
    private Integer checkedItems;
    private Integer abnormalItems;
    private String checkResult;
    private String itemResults;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}