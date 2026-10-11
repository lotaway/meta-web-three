package com.metawebthree.mes.infrastructure.persistence.dataobject.labor;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("mes_work_center_assignment")
public class WorkCenterAssignmentDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long operatorId;
    private String workCenterId;
    private String workCenterName;
    private LocalDate startDate;
    private LocalDate endDate;
    private String shiftType;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
