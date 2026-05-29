package com.metawebthree.finance.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("finance_account_subject")
public class AccountSubjectDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String subjectCode;
    private String subjectName;
    private String direction;
    private Long parentId;
    private Integer level;
    private String status;
    private BigDecimal balance;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}