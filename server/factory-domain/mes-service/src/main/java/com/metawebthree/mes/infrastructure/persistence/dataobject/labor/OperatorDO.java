package com.metawebthree.mes.infrastructure.persistence.dataobject.labor;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_operator")
public class OperatorDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String operatorCode;
    private String operatorName;
    private String department;
    private String jobTitle;
    private String shiftGroup;
    private String status;
    private String phone;
    private String email;
    private String idCardNo;
    private LocalDateTime hireDate;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
