package com.metaweb.datasource.pipeline.controller;

import com.metaweb.datasource.pipeline.model.DataLineageModels.*;
import com.metaweb.datasource.pipeline.service.DataLineageService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/lineage")
public class DataLineageController {

    @Autowired
    private DataLineageService lineageService;

    @PostMapping("/initialize")
    public ResponseEntity<Map<String, Object>> initialize() {
        lineageService.autoRegisterPipelineLineage();
        return ResponseEntity.ok(Map.of("status", "ok", "message", "Pipeline lineage registered"));
    }

    @GetMapping("/graph")
    public ResponseEntity<LineageGraph> getLineageGraph() {
        return ResponseEntity.ok(lineageService.getLineageGraph());
    }

    @GetMapping("/upstream/{nodeId}")
    public ResponseEntity<LineageGraph> getUpstreamLineage(@PathVariable String nodeId) {
        return ResponseEntity.ok(lineageService.getUpstreamLineage(nodeId));
    }

    @GetMapping("/downstream/{nodeId}")
    public ResponseEntity<LineageGraph> getDownstreamLineage(@PathVariable String nodeId) {
        return ResponseEntity.ok(lineageService.getDownstreamLineage(nodeId));
    }

    @GetMapping("/impact/{nodeId}")
    public ResponseEntity<ImpactAnalysis> analyzeImpact(@PathVariable String nodeId) {
        return ResponseEntity.ok(lineageService.analyzeImpact(nodeId));
    }

    @GetMapping("/column/{nodeId}")
    public ResponseEntity<List<ColumnLineage>> getColumnLineage(@PathVariable String nodeId) {
        return ResponseEntity.ok(lineageService.getColumnLineage(nodeId));
    }

    @PostMapping("/node")
    public ResponseEntity<LineageNode> registerNode(@RequestBody RegisterNodeRequest request) {
        return ResponseEntity.ok(lineageService.registerNode(
                request.getNodeId(), request.getName(), request.getType(), request.getSystem(),
                request.getDatabase(), request.getTable(), request.getFields(), request.getMetadata()
        ));
    }

    @PostMapping("/edge")
    public ResponseEntity<LineageEdge> registerEdge(@RequestBody RegisterEdgeRequest request) {
        return ResponseEntity.ok(lineageService.registerEdge(
                request.getSourceNodeId(), request.getTargetNodeId(), request.getEdgeType(),
                request.getTransformation(), request.getMetadata()
        ));
    }

    @PostMapping("/column-lineage")
    public ResponseEntity<Map<String, Object>> registerColumnLineage(@RequestBody RegisterColumnLineageRequest request) {
        lineageService.registerColumnLineage(
                request.getSourceNodeId(), request.getSourceColumn(),
                request.getTargetNodeId(), request.getTargetColumn(),
                request.getTransformation()
        );
        return ResponseEntity.ok(Map.of("status", "ok"));
    }

    @lombok.Data
    public static class RegisterNodeRequest {
        private String nodeId;
        private String name;
        private NodeType type;
        private String system;
        private String database;
        private String table;
        private List<String> fields;
        private Map<String, String> metadata;
    }

    @lombok.Data
    public static class RegisterEdgeRequest {
        private String sourceNodeId;
        private String targetNodeId;
        private EdgeType edgeType;
        private String transformation;
        private Map<String, String> metadata;
    }

    @lombok.Data
    public static class RegisterColumnLineageRequest {
        private String sourceNodeId;
        private String sourceColumn;
        private String targetNodeId;
        private String targetColumn;
        private String transformation;
    }
}
