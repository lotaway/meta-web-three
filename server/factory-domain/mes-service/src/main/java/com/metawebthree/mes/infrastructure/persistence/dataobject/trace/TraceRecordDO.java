package com.metawebthree.mes.infrastructure.persistence.dataobject.trace;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data
@TableName(value = "mes_trace_record", autoResultMap = true)
public class TraceRecordDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String traceCode;
    private String traceType;
    private String productCode;
    private String productName;
    private String batchNo;
    private String sn;
    private String sourceTraceCode;
    private String source;
    private String workOrderNo;
    private String equipmentCode;
    private String equipmentName;
    private String operatorCode;
    private String operatorName;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> relations;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> contextJson;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
