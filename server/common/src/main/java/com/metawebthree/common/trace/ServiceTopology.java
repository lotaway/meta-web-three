package com.metawebthree.common.trace;

import java.util.ArrayList;
import java.util.List;

/**
 * Service topology graph containing nodes and edges
 * Represents the runtime service dependency relationships
 */
public class ServiceTopology {
    private List<ServiceNode> nodes = new ArrayList<>();
    private List<ServiceEdge> edges = new ArrayList<>();
    private long generatedAt;

    public List<ServiceNode> getNodes() {
        return nodes;
    }

    public void setNodes(List<ServiceNode> nodes) {
        this.nodes = nodes;
    }

    public List<ServiceEdge> getEdges() {
        return edges;
    }

    public void setEdges(List<ServiceEdge> edges) {
        this.edges = edges;
    }

    public long getGeneratedAt() {
        return generatedAt;
    }

    public void setGeneratedAt(long generatedAt) {
        this.generatedAt = generatedAt;
    }

    public void addNode(ServiceNode node) {
        nodes.add(node);
    }

    public void addEdge(ServiceEdge edge) {
        edges.add(edge);
    }
}
