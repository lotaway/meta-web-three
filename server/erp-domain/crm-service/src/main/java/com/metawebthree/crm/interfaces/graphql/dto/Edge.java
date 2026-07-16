package com.metawebthree.crm.interfaces.graphql.dto;

public class Edge<T> {
    private String cursor;
    private T node;

    public Edge(String cursor, T node) {
        this.cursor = cursor;
        this.node = node;
    }

    public String getCursor() { return cursor; }
    public T getNode() { return node; }
}
