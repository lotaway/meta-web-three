package com.metawebthree.inventory.application.dto.stockcheck;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class StockCheckPlanDTO {
    private Long id;
    private String planNo;
    private String planName;
    private String checkType;
    private Long warehouseId;
    private String warehouseName;
    private String status;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private String creator;
    private LocalDateTime createTime;
    private String remark;

    private List<StockCheckPlanDetailDTO> details;

    private Integer totalSkus;
    private Integer completedSkus;
    private Integer differenceCount;
}