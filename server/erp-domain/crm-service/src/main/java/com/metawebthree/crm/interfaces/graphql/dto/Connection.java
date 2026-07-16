package com.metawebthree.crm.interfaces.graphql.dto;

import java.util.List;

public class Connection<T> {
    private List<Edge<T>> edges;
    private int totalCount;
    private PageInfo pageInfo;

    public Connection(List<Edge<T>> edges, int totalCount, PageInfo pageInfo) {
        this.edges = edges;
        this.totalCount = totalCount;
        this.pageInfo = pageInfo;
    }

    public List<Edge<T>> getEdges() { return edges; }
    public int getTotalCount() { return totalCount; }
    public PageInfo getPageInfo() { return pageInfo; }
}
