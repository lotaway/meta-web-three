package com.metawebthree.rma.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class RmaInspection {
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String inspector;
    private LocalDateTime inspectionDate;
    private String result;
    private String conclusion;
    private Integer totalInspected;
    private Integer totalPassed;
    private Integer totalFailed;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
