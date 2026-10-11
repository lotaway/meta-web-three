package com.metawebthree.order.domain.model;

import com.baomidou.mybatisplus.annotation.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Saga transaction instance entity.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_saga_instance")
public class SagaInstance {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String sagaId;

    private String bizId;

    private String sagaType;

    private String status;

    private String currentStep;

    private LocalDateTime startTime;

    private LocalDateTime endTime;

    private String errorMessage;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    /**
     * Saga status enum
     */
    public static final class Status {
        public static final String RUNNING = "RUNNING";
        public static final String COMPLETED = "COMPLETED";
        public static final String COMPENSATED = "COMPENSATED";
        public static final String FAILED = "FAILED";
    }

    /**
     * Saga type enum
     */
    public static final class Type {
        public static final String ORDER_PAYMENT_SAGA = "ORDER_PAYMENT_SAGA";
    }
}
