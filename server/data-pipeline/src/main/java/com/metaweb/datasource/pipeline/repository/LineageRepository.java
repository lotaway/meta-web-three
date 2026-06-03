package com.metaweb.datasource.pipeline.repository;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metaweb.datasource.pipeline.model.DataLineageModels.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Repository
public class LineageRepository {

    @Autowired
    @Qualifier("clickHouseJdbcTemplate")
    private JdbcTemplate clickHouseJdbcTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static final String TABLE_NODE = "meta_web_analytics.lineage_node";
    private static final String TABLE_EDGE = "meta_web_analytics.lineage_edge";
    private static final String TABLE_COLUMN = "meta_web_analytics.column_lineage";

    private static final String NODE_COLUMNS = "node_id, name, type, system, database_name, table_name, fields, metadata, created_at";
    private static final String EDGE_COLUMNS = "edge_id, source_node_id, target_node_id, edge_type, transformation, metadata, created_at";
    private static final String COLUMN_COLUMNS = "source_node_id, source_column, target_node_id, target_column, transformation";

    public void initTables() {
        try {
            createNodeTable();
            createEdgeTable();
            createColumnLineageTable();
            log.info("Lineage tables initialized successfully");
        } catch (Exception e) {
            log.warn("Failed to initialize lineage tables: {}", e.getMessage());
        }
    }

    private void createNodeTable() {
        clickHouseJdbcTemplate.execute("""
            CREATE TABLE IF NOT EXISTS """ + TABLE_NODE + """ (
                node_id String, name String, type String, system String,
                database_name String, table_name String, fields String,
                metadata String, created_at DateTime
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (node_id)
            SETTINGS index_granularity = 8192
        """);
    }

    private void createEdgeTable() {
        clickHouseJdbcTemplate.execute("""
            CREATE TABLE IF NOT EXISTS """ + TABLE_EDGE + """ (
                edge_id String, source_node_id String, target_node_id String,
                edge_type String, transformation String, metadata String, created_at DateTime
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (edge_id)
            SETTINGS index_granularity = 8192
        """);
    }

    private void createColumnLineageTable() {
        clickHouseJdbcTemplate.execute("""
            CREATE TABLE IF NOT EXISTS """ + TABLE_COLUMN + """ (
                source_node_id String, source_column String,
                target_node_id String, target_column String, transformation String
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (source_node_id, source_column, target_node_id, target_column)
            SETTINGS index_granularity = 8192
        """);
    }

    public void saveNode(LineageNode node) {
        try {
            String fieldsJson = objectMapper.writeValueAsString(node.getFields() != null ? node.getFields() : List.of());
            String metadataJson = objectMapper.writeValueAsString(node.getMetadata() != null ? node.getMetadata() : Map.of());
            clickHouseJdbcTemplate.update(
                    "INSERT INTO " + TABLE_NODE + " (" + NODE_COLUMNS + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    node.getNodeId(), node.getName(), node.getType().name(), node.getSystem(),
                    node.getDatabase(), node.getTable(), fieldsJson, metadataJson,
                    node.getCreatedAt() != null ? node.getCreatedAt().format(FMT) : LocalDateTime.now().format(FMT)
            );
        } catch (Exception e) {
            log.error("Failed to save lineage node: {}", node.getNodeId(), e);
        }
    }

    public List<LineageNode> getAllNodes() {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT * FROM " + TABLE_NODE + " FINAL ORDER BY node_id",
                    (rs, rowNum) -> mapToLineageNode(rs)
            );
        } catch (Exception e) {
            log.error("Failed to query lineage nodes", e);
            return List.of();
        }
    }

    public List<LineageNode> getNodesByType(NodeType type) {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT * FROM " + TABLE_NODE + " FINAL WHERE type = ? ORDER BY node_id",
                    (rs, rowNum) -> mapToLineageNode(rs),
                    type.name()
            );
        } catch (Exception e) {
            log.error("Failed to query lineage nodes by type", e);
            return List.of();
        }
    }

    public void saveEdge(LineageEdge edge) {
        try {
            String metadataJson = objectMapper.writeValueAsString(edge.getMetadata() != null ? edge.getMetadata() : Map.of());
            clickHouseJdbcTemplate.update(
                    "INSERT INTO " + TABLE_EDGE + " (" + EDGE_COLUMNS + ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    edge.getEdgeId(), edge.getSourceNodeId(), edge.getTargetNodeId(),
                    edge.getEdgeType().name(), edge.getTransformation(), metadataJson,
                    edge.getCreatedAt() != null ? edge.getCreatedAt().format(FMT) : LocalDateTime.now().format(FMT)
            );
        } catch (Exception e) {
            log.error("Failed to save lineage edge: {}", edge.getEdgeId(), e);
        }
    }

    public List<LineageEdge> getAllEdges() {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT * FROM " + TABLE_EDGE + " FINAL ORDER BY edge_id",
                    (rs, rowNum) -> LineageEdge.builder()
                            .edgeId(rs.getString("edge_id"))
                            .sourceNodeId(rs.getString("source_node_id"))
                            .targetNodeId(rs.getString("target_node_id"))
                            .edgeType(EdgeType.valueOf(rs.getString("edge_type")))
                            .transformation(rs.getString("transformation"))
                            .metadata(readMapJson(rs.getString("metadata")))
                            .createdAt(rs.getTimestamp("created_at").toLocalDateTime())
                            .build()
            );
        } catch (Exception e) {
            log.error("Failed to query lineage edges", e);
            return List.of();
        }
    }

    public List<LineageNode> getUpstreamNodes(String nodeId) {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT n.* FROM " + TABLE_NODE + " n FINAL " +
                            "INNER JOIN " + TABLE_EDGE + " e FINAL ON n.node_id = e.source_node_id " +
                            "WHERE e.target_node_id = ?",
                    (rs, rowNum) -> mapToLineageNode(rs),
                    nodeId
            );
        } catch (Exception e) {
            log.error("Failed to query upstream nodes for: {}", nodeId, e);
            return List.of();
        }
    }

    public List<LineageNode> getDownstreamNodes(String nodeId) {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT n.* FROM " + TABLE_NODE + " n FINAL " +
                            "INNER JOIN " + TABLE_EDGE + " e FINAL ON n.node_id = e.target_node_id " +
                            "WHERE e.source_node_id = ?",
                    (rs, rowNum) -> mapToLineageNode(rs),
                    nodeId
            );
        } catch (Exception e) {
            log.error("Failed to query downstream nodes for: {}", nodeId, e);
            return List.of();
        }
    }

    public void saveColumnLineage(ColumnLineage columnLineage) {
        try {
            clickHouseJdbcTemplate.update(
                    "INSERT INTO " + TABLE_COLUMN + " (" + COLUMN_COLUMNS + ") VALUES (?, ?, ?, ?, ?)",
                    columnLineage.getSourceNodeId(), columnLineage.getSourceColumn(),
                    columnLineage.getTargetNodeId(), columnLineage.getTargetColumn(),
                    columnLineage.getTransformation()
            );
        } catch (Exception e) {
            log.error("Failed to save column lineage", e);
        }
    }

    public List<ColumnLineage> getColumnLineage(String nodeId) {
        try {
            return clickHouseJdbcTemplate.query(
                    "SELECT * FROM " + TABLE_COLUMN + " FINAL WHERE source_node_id = ? OR target_node_id = ?",
                    (rs, rowNum) -> ColumnLineage.builder()
                            .sourceNodeId(rs.getString("source_node_id"))
                            .sourceColumn(rs.getString("source_column"))
                            .targetNodeId(rs.getString("target_node_id"))
                            .targetColumn(rs.getString("target_column"))
                            .transformation(rs.getString("transformation"))
                            .build(),
                    nodeId, nodeId
            );
        } catch (Exception e) {
            log.error("Failed to query column lineage for: {}", nodeId, e);
            return List.of();
        }
    }

    private LineageNode mapToLineageNode(java.sql.ResultSet rs) throws java.sql.SQLException {
        return LineageNode.builder()
                .nodeId(rs.getString("node_id"))
                .name(rs.getString("name"))
                .type(NodeType.valueOf(rs.getString("type")))
                .system(rs.getString("system"))
                .database(rs.getString("database_name"))
                .table(rs.getString("table_name"))
                .fields(readListJson(rs.getString("fields")))
                .metadata(readMapJson(rs.getString("metadata")))
                .createdAt(rs.getTimestamp("created_at") != null ? rs.getTimestamp("created_at").toLocalDateTime() : null)
                .build();
    }

    private List<String> readListJson(String json) {
        try {
            return objectMapper.readValue(json, List.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse JSON list: " + json, e);
        }
    }

    private Map<String, String> readMapJson(String json) {
        try {
            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse JSON map: " + json, e);
        }
    }
}
