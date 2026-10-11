package com.metawebthree.rma.application.dto;

import lombok.Data;

@Data
public class RecordInspectionRequest {
    private String inspector;
    private String result;
    private String conclusion;
    private Integer totalInspected;
    private Integer totalPassed;
    private Integer totalFailed;
    private String remark;
}
