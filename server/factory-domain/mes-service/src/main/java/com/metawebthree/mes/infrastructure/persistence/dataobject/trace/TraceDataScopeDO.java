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
@TableName(value = "mes_trace_data_scope", autoResultMap = true)
public class TraceDataScopeDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String scopeCode;
    private String scopeName;
    private String scopeType;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> items;

    private Boolean isDefault;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
