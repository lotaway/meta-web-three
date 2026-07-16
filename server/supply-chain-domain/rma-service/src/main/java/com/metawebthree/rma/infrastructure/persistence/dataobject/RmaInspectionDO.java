package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("rma_inspection")
public class RmaInspectionDO {
    @TableId(type = IdType.ASSIGN_ID)
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
