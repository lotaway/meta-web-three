package com.metawebthree.crm.interfaces.graphql.dto;

public class PageInfo {
    private boolean hasNextPage;
    private boolean hasPreviousPage;
    private String startCursor;
    private String endCursor;

    public PageInfo(boolean hasNextPage, String endCursor) {
        this.hasNextPage = hasNextPage;
        this.hasPreviousPage = false;
        this.endCursor = endCursor;
    }

    public boolean isHasNextPage() { return hasNextPage; }
    public boolean isHasPreviousPage() { return hasPreviousPage; }
    public String getStartCursor() { return startCursor; }
    public String getEndCursor() { return endCursor; }
}
