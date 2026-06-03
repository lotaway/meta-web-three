package com.metaweb.datasource.pipeline.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class DataLineageModels {

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LineageNode {
        private String nodeId;
        private String name;
        private NodeType type;
        private String system;
        private String database;
        private String table;
        private List<String> fields;
        private Map<String, String> metadata;
        private LocalDateTime createdAt;
    }

    public enum NodeType {
        SOURCE,
        TRANSFORMATION,
        SINK,
        CONSUMER
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LineageEdge {
        private String edgeId;
        private String sourceNodeId;
        private String targetNodeId;
        private EdgeType edgeType;
        private String transformation;
        private Map<String, String> metadata;
        private LocalDateTime createdAt;
    }

    public enum EdgeType {
        READS_FROM,
        WRITES_TO,
        TRANSFORMS_TO,
        DERIVED_FROM
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LineageGraph {
        private String graphId;
        private String name;
        private List<LineageNode> nodes;
        private List<LineageEdge> edges;
        private LocalDateTime generatedAt;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ColumnLineage {
        private String sourceNodeId;
        private String sourceColumn;
        private String targetNodeId;
        private String targetColumn;
        private String transformation;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ImpactAnalysis {
        private String sourceNodeId;
        private List<ImpactedNode> impactedNodes;
        private List<String> impactedPaths;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ImpactedNode {
        private String nodeId;
        private String name;
        private NodeType type;
        private int depth;
        private String path;
    }
}
