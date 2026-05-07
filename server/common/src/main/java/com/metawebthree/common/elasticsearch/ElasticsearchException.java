package com.metawebthree.common.elasticsearch;

public class ElasticsearchException extends RuntimeException {
    public ElasticsearchException(String message) {
        super(message);
    }

    public ElasticsearchException(String message, Throwable cause) {
        super(message, cause);
    }
}
