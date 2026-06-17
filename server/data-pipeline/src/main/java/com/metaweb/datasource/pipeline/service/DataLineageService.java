package com.metaweb.datasource.pipeline.service;

import com.metaweb.datasource.pipeline.model.DataLineageModels.*;
import com.metaweb.datasource.pipeline.repository.LineageRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class DataLineageService {

    @Autowired
    private LineageRepository lineageRepository;

    public LineageNode registerNode(String nodeId, String name, NodeType type, String system,
                                      String database, String table, List<String> fields,
                                      Map<String, String> metadata) {
        LineageNode node = LineageNode.builder()
                .nodeId(nodeId).name(name).type(type)
                .system(system).database(database).table(table)
                .fields(fields).metadata(metadata)
                .createdAt(LocalDateTime.now())
                .build();
        lineageRepository.saveNode(node);
        log.info("Registered lineage node: {} ({})", nodeId, type);
        return node;
    }

    public LineageEdge registerEdge(String sourceNodeId, String targetNodeId, EdgeType edgeType,
                                      String transformation, Map<String, String> metadata) {
        String edgeId = sourceNodeId + "->" + targetNodeId;
        LineageEdge edge = LineageEdge.builder()
                .edgeId(edgeId).sourceNodeId(sourceNodeId).targetNodeId(targetNodeId)
                .edgeType(edgeType).transformation(transformation).metadata(metadata)
                .createdAt(LocalDateTime.now())
                .build();
        lineageRepository.saveEdge(edge);
        log.info("Registered lineage edge: {}", edgeId);
        return edge;
    }

    public void registerColumnLineage(String sourceNodeId, String sourceColumn,
                                        String targetNodeId, String targetColumn,
                                        String transformation) {
        ColumnLineage cl = ColumnLineage.builder()
                .sourceNodeId(sourceNodeId).sourceColumn(sourceColumn)
                .targetNodeId(targetNodeId).targetColumn(targetColumn)
                .transformation(transformation)
                .build();
        lineageRepository.saveColumnLineage(cl);
    }

    public LineageGraph getLineageGraph() {
        return LineageGraph.builder()
                .graphId("full-lineage").name("Full Data Lineage Graph")
                .nodes(lineageRepository.getAllNodes())
                .edges(lineageRepository.getAllEdges())
                .generatedAt(LocalDateTime.now())
                .build();
    }

    public LineageGraph getUpstreamLineage(String nodeId) {
        List<LineageNode> allNodes = lineageRepository.getAllNodes();
        List<LineageEdge> allEdges = lineageRepository.getAllEdges();
        Map<String, List<LineageEdge>> upstreamEdges = allEdges.stream()
                .collect(Collectors.groupingBy(LineageEdge::getTargetNodeId));
        Set<String> visitedNodes = new HashSet<>();
        visitedNodes.add(nodeId);
        List<LineageEdge> resultEdges = bfsEdges(nodeId, upstreamEdges, visitedNodes);
        List<LineageNode> resultNodes = collectNodes(allNodes, visitedNodes);
        return LineageGraph.builder()
                .graphId("upstream-" + nodeId).name("Upstream Lineage for " + nodeId)
                .nodes(resultNodes).edges(resultEdges)
                .generatedAt(LocalDateTime.now())
                .build();
    }

    public LineageGraph getDownstreamLineage(String nodeId) {
        List<LineageNode> allNodes = lineageRepository.getAllNodes();
        List<LineageEdge> allEdges = lineageRepository.getAllEdges();
        Map<String, List<LineageEdge>> downstreamEdges = allEdges.stream()
                .collect(Collectors.groupingBy(LineageEdge::getSourceNodeId));
        Set<String> visitedNodes = new HashSet<>();
        visitedNodes.add(nodeId);
        List<LineageEdge> resultEdges = bfsEdges(nodeId, downstreamEdges, visitedNodes);
        List<LineageNode> resultNodes = collectNodes(allNodes, visitedNodes);
        return LineageGraph.builder()
                .graphId("downstream-" + nodeId).name("Downstream Lineage for " + nodeId)
                .nodes(resultNodes).edges(resultEdges)
                .generatedAt(LocalDateTime.now())
                .build();
    }

    public ImpactAnalysis analyzeImpact(String sourceNodeId) {
        LineageGraph downstream = getDownstreamLineage(sourceNodeId);
        Map<String, LineageNode> nodeMap = downstream.getNodes().stream()
                .collect(Collectors.toMap(LineageNode::getNodeId, n -> n));
        Map<String, List<LineageEdge>> downstreamEdges = downstream.getEdges().stream()
                .collect(Collectors.groupingBy(LineageEdge::getSourceNodeId));
        List<ImpactedNode> impactedNodes = new ArrayList<>();
        List<String> impactedPaths = new ArrayList<>();
        bfsImpact(sourceNodeId, downstreamEdges, nodeMap, impactedNodes, impactedPaths);
        return ImpactAnalysis.builder()
                .sourceNodeId(sourceNodeId)
                .impactedNodes(impactedNodes)
                .impactedPaths(impactedPaths)
                .build();
    }

    public List<ColumnLineage> getColumnLineage(String nodeId) {
        return lineageRepository.getColumnLineage(nodeId);
    }

    public void autoRegisterPipelineLineage() {
        registerSourceNodes();
        registerEtlNodes();
        registerSinkNodes();
        registerConsumerNodes();
        registerEdges();
        registerColumnLineageRelations();
        log.info("Auto-registered pipeline lineage with 11 nodes and edges");
    }

    private List<LineageEdge> bfsEdges(String startNodeId, Map<String, List<LineageEdge>> edgeMap,
                                        Set<String> visitedNodes) {
        List<LineageEdge> resultEdges = new ArrayList<>();
        Set<String> visitedEdges = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.add(startNodeId);
        while (!queue.isEmpty()) {
            String current = queue.poll();
            for (LineageEdge edge : edgeMap.getOrDefault(current, List.of())) {
                if (visitedEdges.add(edge.getEdgeId())) {
                    resultEdges.add(edge);
                    String neighbor = edge.getSourceNodeId();
                    if (visitedNodes.add(neighbor)) {
                        queue.add(neighbor);
                    }
                }
            }
        }
        return resultEdges;
    }

    private List<LineageNode> collectNodes(List<LineageNode> allNodes, Set<String> nodeIds) {
        Map<String, LineageNode> nodeMap = allNodes.stream()
                .collect(Collectors.toMap(LineageNode::getNodeId, n -> n));
        List<LineageNode> result = new ArrayList<>();
        for (String nid : nodeIds) {
            if (nodeMap.containsKey(nid)) {
                result.add(nodeMap.get(nid));
            }
        }
        return result;
    }

    private void bfsImpact(String sourceNodeId, Map<String, List<LineageEdge>> downstreamEdges,
                            Map<String, LineageNode> nodeMap,
                            List<ImpactedNode> impactedNodes, List<String> impactedPaths) {
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> depthMap = new HashMap<>();
        Map<String, String> pathMap = new HashMap<>();
        queue.add(sourceNodeId);
        depthMap.put(sourceNodeId, 0);
        pathMap.put(sourceNodeId, sourceNodeId);
        while (!queue.isEmpty()) {
            String current = queue.poll();
            int depth = depthMap.get(current);
            for (LineageEdge edge : downstreamEdges.getOrDefault(current, List.of())) {
                String target = edge.getTargetNodeId();
                if (depthMap.containsKey(target)) continue;
                String newPath = pathMap.get(current) + " -> " + target;
                depthMap.put(target, depth + 1);
                pathMap.put(target, newPath);
                queue.add(target);
                if (nodeMap.containsKey(target)) {
                    impactedNodes.add(ImpactedNode.builder()
                            .nodeId(target).name(nodeMap.get(target).getName())
                            .type(nodeMap.get(target).getType())
                            .depth(depth + 1).path(newPath).build());
                }
                impactedPaths.add(newPath);
            }
        }
    }

    private void registerSourceNodes() {
        registerNode("kafka-order-events", "Order Events Topic", NodeType.SOURCE, "kafka", null,
                "meta-web-order-events", List.of("orderId", "userId", "totalAmount", "status", "eventTime"),
                Map.of("partitions", "3", "retention", "7d"));
        registerNode("kafka-inventory-events", "Inventory Events Topic", NodeType.SOURCE, "kafka", null,
                "meta-web-inventory-events", List.of("productId", "quantity", "warehouseId", "eventTime"),
                Map.of("partitions", "3", "retention", "7d"));
        registerNode("kafka-user-behavior-events", "User Behavior Events Topic", NodeType.SOURCE, "kafka", null,
                "meta-web-user-behavior-events", List.of("userId", "sessionId", "eventType", "pageUrl", "eventTime"),
                Map.of("partitions", "3", "retention", "7d"));
    }

    private void registerEtlNodes() {
        registerNode("etl-order", "Order ETL", NodeType.TRANSFORMATION, "data-pipeline", null,
                "OrderEventConsumer+EtlService", List.of("eventId", "eventType", "orderId", "userId", "totalAmount", "yearMonth", "dayOfWeek", "hourOfDay"),
                Map.of("input", "kafka-order-events", "output", "clickhouse-order-analytics"));
        registerNode("etl-inventory", "Inventory ETL", NodeType.TRANSFORMATION, "data-pipeline", null,
                "InventoryEventConsumer+EtlService", List.of("eventId", "eventType", "productId", "productName", "quantity", "availableQty"),
                Map.of("input", "kafka-inventory-events", "output", "clickhouse-inventory-analytics"));
        registerNode("etl-user-behavior", "User Behavior ETL", NodeType.TRANSFORMATION, "data-pipeline", null,
                "UserBehaviorEventConsumer+EtlService", List.of("eventId", "eventType", "userId", "sessionId", "browserFamily"),
                Map.of("input", "kafka-user-behavior-events", "output", "clickhouse-user-behavior-analytics"));
    }

    private void registerSinkNodes() {
        registerNode("clickhouse-order-analytics", "Order Analytics Table", NodeType.SINK, "clickhouse", "meta_web_analytics",
                "order_analytics", List.of("eventId", "eventType", "orderId", "userId", "totalAmount", "status", "eventTime", "yearMonth", "dayOfWeek", "hourOfDay"),
                Map.of("engine", "MergeTree", "partition", "toYYYYMM(eventTime)"));
        registerNode("clickhouse-inventory-analytics", "Inventory Analytics Table", NodeType.SINK, "clickhouse", "meta_web_analytics",
                "inventory_analytics", List.of("eventId", "eventType", "productId", "quantity", "availableQty", "eventTime"),
                Map.of("engine", "MergeTree", "partition", "toYYYYMM(event_time)"));
        registerNode("clickhouse-user-behavior-analytics", "User Behavior Analytics Table", NodeType.SINK, "clickhouse", "meta_web_analytics",
                "user_behavior_analytics", List.of("eventId", "eventType", "userId", "sessionId", "deviceType", "eventTime"),
                Map.of("engine", "MergeTree", "partition", "toYYYYMM(event_time)"));
    }

    private void registerConsumerNodes() {
        registerNode("analytics-api", "Analytics REST API", NodeType.CONSUMER, "data-pipeline", null,
                "AnalyticsController", List.of("orderAnalytics", "inventoryAnalytics", "userBehaviorAnalytics", "dashboard"),
                Map.of("basePath", "/api/analytics"));
        registerNode("olap-api", "OLAP Query API", NodeType.CONSUMER, "data-pipeline", null,
                "OlapController", List.of("drillDown", "rollUp", "slice", "dice", "pivot", "salesFunnel"),
                Map.of("basePath", "/api/olap"));
        registerNode("dashboard-ws", "Dashboard WebSocket", NodeType.CONSUMER, "data-pipeline", null,
                "DashboardWebSocketHandler", List.of("metrics", "orderAlert", "inventoryAlert"),
                Map.of("endpoint", "/ws/dashboard"));
    }

    private void registerEdges() {
        registerEdge("kafka-order-events", "etl-order", EdgeType.READS_FROM, "Consume & Transform", null);
        registerEdge("kafka-inventory-events", "etl-inventory", EdgeType.READS_FROM, "Consume & Transform", null);
        registerEdge("kafka-user-behavior-events", "etl-user-behavior", EdgeType.READS_FROM, "Consume & Transform", null);
        registerEdge("etl-order", "clickhouse-order-analytics", EdgeType.WRITES_TO, "Insert Batch", null);
        registerEdge("etl-inventory", "clickhouse-inventory-analytics", EdgeType.WRITES_TO, "Insert Batch", null);
        registerEdge("etl-user-behavior", "clickhouse-user-behavior-analytics", EdgeType.WRITES_TO, "Insert Batch", null);
        registerEdge("clickhouse-order-analytics", "analytics-api", EdgeType.READS_FROM, "SQL Query", null);
        registerEdge("clickhouse-inventory-analytics", "analytics-api", EdgeType.READS_FROM, "SQL Query", null);
        registerEdge("clickhouse-user-behavior-analytics", "analytics-api", EdgeType.READS_FROM, "SQL Query", null);
        registerEdge("clickhouse-order-analytics", "olap-api", EdgeType.READS_FROM, "OLAP Query", null);
        registerEdge("clickhouse-inventory-analytics", "olap-api", EdgeType.READS_FROM, "OLAP Query", null);
        registerEdge("clickhouse-user-behavior-analytics", "olap-api", EdgeType.READS_FROM, "OLAP Query", null);
        registerEdge("clickhouse-order-analytics", "dashboard-ws", EdgeType.READS_FROM, "Scheduled Push", null);
        registerEdge("clickhouse-inventory-analytics", "dashboard-ws", EdgeType.READS_FROM, "Alert Push", null);
    }

    private void registerColumnLineageRelations() {
        registerColumnLineage("kafka-order-events", "orderId", "etl-order", "orderId", "direct");
        registerColumnLineage("kafka-order-events", "userId", "etl-order", "userId", "direct");
        registerColumnLineage("kafka-order-events", "totalAmount", "etl-order", "totalAmount", "direct");
        registerColumnLineage("kafka-order-events", "eventTime", "etl-order", "yearMonth", "toYYYYMM()");
        registerColumnLineage("kafka-order-events", "eventTime", "etl-order", "dayOfWeek", "getDayOfWeek()");
        registerColumnLineage("kafka-order-events", "eventTime", "etl-order", "hourOfDay", "getHour()");
        registerColumnLineage("etl-order", "orderId", "clickhouse-order-analytics", "order_id", "direct");
        registerColumnLineage("etl-order", "totalAmount", "clickhouse-order-analytics", "total_amount", "direct");
        registerColumnLineage("clickhouse-order-analytics", "total_amount", "olap-api", "total_amount_sum", "sum()");
        registerColumnLineage("clickhouse-order-analytics", "order_id", "olap-api", "unique_orders", "uniqExact()");
        registerColumnLineage("clickhouse-order-analytics", "total_amount", "dashboard-ws", "total_amount", "sum()");
    }
}
