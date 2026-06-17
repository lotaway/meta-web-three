package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("column_lineage")
public class ColumnLineageDO {
    @TableField("source_node_id")
    private String sourceNodeId;
    @TableField("source_column")
    private String sourceColumn;
    @TableField("target_node_id")
    private String targetNodeId;
    @TableField("target_column")
    private String targetColumn;
    private String transformation;
}
