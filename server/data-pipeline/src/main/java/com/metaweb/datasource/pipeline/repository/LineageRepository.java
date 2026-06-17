package com.metaweb.datasource.pipeline.repository;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metaweb.datasource.pipeline.model.DataLineageModels.*;
import com.metaweb.datasource.pipeline.repository.entity.ColumnLineageDO;
import com.metaweb.datasource.pipeline.repository.entity.LineageEdgeDO;
import com.metaweb.datasource.pipeline.repository.entity.LineageNodeDO;
import com.metaweb.datasource.pipeline.repository.mapper.ColumnLineageMapper;
import com.metaweb.datasource.pipeline.repository.mapper.LineageEdgeMapper;
import com.metaweb.datasource.pipeline.repository.mapper.LineageNodeMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Repository
public class LineageRepository {

    @Autowired
    private LineageNodeMapper nodeMapper;

    @Autowired
    private LineageEdgeMapper edgeMapper;

    @Autowired
    private ColumnLineageMapper columnMapper;

    @Autowired
    private ObjectMapper objectMapper;

    private static final TypeReference<List<String>> LIST_STRING_TYPE = new TypeReference<List<String>>() {};
    private static final TypeReference<Map<String, String>> MAP_STRING_TYPE = new TypeReference<Map<String, String>>() {};

    public void saveNode(LineageNode node) {
        try {
            nodeMapper.insert(toNodeDO(node));
        } catch (Exception e) {
            log.error("Failed to save lineage node: {}", node.getNodeId(), e);
        }
    }

    public List<LineageNode> getAllNodes() {
        try {
            return nodeMapper.selectList(null).stream()
                    .map(this::toLineageNode)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query lineage nodes", e);
            return List.of();
        }
    }

    public List<LineageNode> getNodesByType(NodeType type) {
        try {
            return nodeMapper.selectList(
                    new LambdaQueryWrapper<LineageNodeDO>()
                            .eq(LineageNodeDO::getType, type.name())
            ).stream().map(this::toLineageNode).collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query lineage nodes by type", e);
            return List.of();
        }
    }

    public void saveEdge(LineageEdge edge) {
        try {
            edgeMapper.insert(toEdgeDO(edge));
        } catch (Exception e) {
            log.error("Failed to save lineage edge: {}", edge.getEdgeId(), e);
        }
    }

    public List<LineageEdge> getAllEdges() {
        try {
            return edgeMapper.selectList(null).stream()
                    .map(this::toLineageEdge)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query lineage edges", e);
            return List.of();
        }
    }

    public List<LineageNode> getUpstreamNodes(String nodeId) {
        try {
            return nodeMapper.selectUpstreamNodes(nodeId).stream()
                    .map(this::toLineageNode)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query upstream nodes for: {}", nodeId, e);
            return List.of();
        }
    }

    public List<LineageNode> getDownstreamNodes(String nodeId) {
        try {
            return nodeMapper.selectDownstreamNodes(nodeId).stream()
                    .map(this::toLineageNode)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query downstream nodes for: {}", nodeId, e);
            return List.of();
        }
    }

    public void saveColumnLineage(ColumnLineage columnLineage) {
        try {
            columnMapper.insert(toColumnLineageDO(columnLineage));
        } catch (Exception e) {
            log.error("Failed to save column lineage", e);
        }
    }

    public List<ColumnLineage> getColumnLineage(String nodeId) {
        try {
            return columnMapper.selectList(
                    new LambdaQueryWrapper<ColumnLineageDO>()
                            .eq(ColumnLineageDO::getSourceNodeId, nodeId)
                            .or()
                            .eq(ColumnLineageDO::getTargetNodeId, nodeId)
            ).stream().map(this::toColumnLineage).collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Failed to query column lineage for: {}", nodeId, e);
            return List.of();
        }
    }

    private LineageNodeDO toNodeDO(LineageNode node) {
        LineageNodeDO entity = new LineageNodeDO();
        entity.setNodeId(node.getNodeId());
        entity.setName(node.getName());
        entity.setType(node.getType() != null ? node.getType().name() : null);
        entity.setSystem(node.getSystem());
        entity.setDatabase(node.getDatabase());
        entity.setTable(node.getTable());
        entity.setFields(toJson(node.getFields()));
        entity.setMetadata(toJson(node.getMetadata()));
        entity.setCreatedAt(node.getCreatedAt());
        return entity;
    }

    private LineageNode toLineageNode(LineageNodeDO entity) {
        return LineageNode.builder()
                .nodeId(entity.getNodeId())
                .name(entity.getName())
                .type(entity.getType() != null ? NodeType.valueOf(entity.getType()) : null)
                .system(entity.getSystem())
                .database(entity.getDatabase())
                .table(entity.getTable())
                .fields(fromJson(entity.getFields(), LIST_STRING_TYPE))
                .metadata(fromJson(entity.getMetadata(), MAP_STRING_TYPE))
                .createdAt(entity.getCreatedAt())
                .build();
    }

    private LineageEdgeDO toEdgeDO(LineageEdge edge) {
        LineageEdgeDO entity = new LineageEdgeDO();
        entity.setEdgeId(edge.getEdgeId());
        entity.setSourceNodeId(edge.getSourceNodeId());
        entity.setTargetNodeId(edge.getTargetNodeId());
        entity.setEdgeType(edge.getEdgeType() != null ? edge.getEdgeType().name() : null);
        entity.setTransformation(edge.getTransformation());
        entity.setMetadata(toJson(edge.getMetadata()));
        entity.setCreatedAt(edge.getCreatedAt());
        return entity;
    }

    private LineageEdge toLineageEdge(LineageEdgeDO entity) {
        return LineageEdge.builder()
                .edgeId(entity.getEdgeId())
                .sourceNodeId(entity.getSourceNodeId())
                .targetNodeId(entity.getTargetNodeId())
                .edgeType(entity.getEdgeType() != null ? EdgeType.valueOf(entity.getEdgeType()) : null)
                .transformation(entity.getTransformation())
                .metadata(fromJson(entity.getMetadata(), MAP_STRING_TYPE))
                .createdAt(entity.getCreatedAt())
                .build();
    }

    private ColumnLineageDO toColumnLineageDO(ColumnLineage cl) {
        ColumnLineageDO entity = new ColumnLineageDO();
        entity.setSourceNodeId(cl.getSourceNodeId());
        entity.setSourceColumn(cl.getSourceColumn());
        entity.setTargetNodeId(cl.getTargetNodeId());
        entity.setTargetColumn(cl.getTargetColumn());
        entity.setTransformation(cl.getTransformation());
        return entity;
    }

    private ColumnLineage toColumnLineage(ColumnLineageDO entity) {
        return ColumnLineage.builder()
                .sourceNodeId(entity.getSourceNodeId())
                .sourceColumn(entity.getSourceColumn())
                .targetNodeId(entity.getTargetNodeId())
                .targetColumn(entity.getTargetColumn())
                .transformation(entity.getTransformation())
                .build();
    }

    private String toJson(Object value) {
        try {
            return value != null ? objectMapper.writeValueAsString(value) : null;
        } catch (Exception e) {
            log.warn("Failed to serialize to JSON", e);
            return null;
        }
    }

    private <T> T fromJson(String json, TypeReference<T> type) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, type);
        } catch (Exception e) {
            log.warn("Failed to parse JSON: {}", json, e);
            return null;
        }
    }
}
