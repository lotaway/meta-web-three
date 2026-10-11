package com.metawebthree.order.domain.model;

import com.baomidou.mybatisplus.annotation.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Saga step execution entity.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_saga_step")
public class SagaStep {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String sagaId;

    private String stepName;

    private Integer stepOrder;

    private String serviceName;

    private Boolean compensable;

    private String status;

    private String requestData;

    private String responseData;

    private String compensationData;

    private Integer retryCount;

    private Integer maxRetries;

    private String errorMessage;

    private LocalDateTime startTime;

    private LocalDateTime endTime;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    /**
     * Step status enum
     */
    public static final class Status {
        public static final String PENDING = "PENDING";
        public static final String RUNNING = "RUNNING";
        public static final String COMPLETED = "COMPLETED";
        public static final String COMPENSATED = "COMPENSATED";
        public static final String FAILED = "FAILED";
    }

    /**
     * Step names for ORDER_PAYMENT_SAGA
     */
    public static final class StepName {
        public static final String CREATE_ORDER = "CREATE_ORDER";
        public static final String RESERVE_INVENTORY = "RESERVE_INVENTORY";
        public static final String PROCESS_PAYMENT = "PROCESS_PAYMENT";
        public static final String CONFIRM_INVENTORY = "CONFIRM_INVENTORY";
    }
}
