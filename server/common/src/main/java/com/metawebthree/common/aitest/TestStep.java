package com.metawebthree.common.aitest;

/**
 * A single step in a test scenario
 */
public class TestStep {
    private int order;
    private String action;
    private String endpoint;
    private String method; // GET, POST, PUT, DELETE
    private String requestBody;
    private String expectedStatus;
    private String expectedResponse;
    private long expectedMaxDurationMs;

    public int getOrder() {
        return order;
    }

    public void setOrder(int order) {
        this.order = order;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public String getEndpoint() {
        return endpoint;
    }

    public void setEndpoint(String endpoint) {
        this.endpoint = endpoint;
    }

    public String getMethod() {
        return method;
    }

    public void setMethod(String method) {
        this.method = method;
    }

    public String getRequestBody() {
        return requestBody;
    }

    public void setRequestBody(String requestBody) {
        this.requestBody = requestBody;
    }

    public String getExpectedStatus() {
        return expectedStatus;
    }

    public void setExpectedStatus(String expectedStatus) {
        this.expectedStatus = expectedStatus;
    }

    public String getExpectedResponse() {
        return expectedResponse;
    }

    public void setExpectedResponse(String expectedResponse) {
        this.expectedResponse = expectedResponse;
    }

    public long getExpectedMaxDurationMs() {
        return expectedMaxDurationMs;
    }

    public void setExpectedMaxDurationMs(long expectedMaxDurationMs) {
        this.expectedMaxDurationMs = expectedMaxDurationMs;
    }
}
