package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("lineage_edge")
public class LineageEdgeDO {
    private String edgeId;
    @TableField("source_node_id")
    private String sourceNodeId;
    @TableField("target_node_id")
    private String targetNodeId;
    @TableField("edge_type")
    private String edgeType;
    private String transformation;
    private String metadata;
    @TableField("created_at")
    private LocalDateTime createdAt;
}
