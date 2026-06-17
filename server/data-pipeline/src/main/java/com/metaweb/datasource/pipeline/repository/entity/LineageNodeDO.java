package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("lineage_node")
public class LineageNodeDO {
    private String nodeId;
    private String name;
    private String type;
    private String system;
    @TableField("database_name")
    private String database;
    @TableField("table_name")
    private String table;
    private String fields;
    private String metadata;
    private LocalDateTime createdAt;
}
