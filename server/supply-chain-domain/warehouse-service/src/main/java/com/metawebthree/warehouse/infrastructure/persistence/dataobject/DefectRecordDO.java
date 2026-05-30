package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class DefectRecordDO {
    private Long id;
    private Long inspectionId;
    private Long inspectionItemId;
    private String skuCode;
    private String productName;
    private String batchNo;
    private String defectType;
    private String defectName;
    private String defectDescription;
    private Integer defectQuantity;
    private String defectLevel;
    private String photoUrls;
    private String locationCode;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Integer deleted;
}