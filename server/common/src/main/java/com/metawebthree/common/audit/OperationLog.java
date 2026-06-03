package com.metawebthree.common.audit;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

/**
 * Operation log entity for auditing user actions
 * Records who operated on what data at what time
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("tb_operation_log")
public class OperationLog extends BaseDO {

    @TableId
    private Long id;

    @TableField("user_id")
    private Long userId;

    @TableField("username")
    private String username;

    @TableField("operation")
    private String operation;

    @TableField("method")
    private String method;

    @TableField("params")
    private String params;

    @TableField("ip")
    private String ip;

    @TableField("operation_time")
    private java.time.LocalDateTime operationTime;

    @TableField("execution_time")
    private Long executionTime;

    @TableField("status")
    private String status;

    @TableField("error_message")
    private String errorMessage;

    @TableField("entity_type")
    private String entityType;

    @TableField("entity_id")
    private Long entityId;
}
