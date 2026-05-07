package com.metawebthree.common.elasticsearch;

import java.util.List;
import java.util.Map;

public interface ElasticsearchOperations {
    void indexDocument(String index, String id, Map<String, Object> document);
    <T> void indexObject(String index, String id, T document);
    void deleteDocument(String index, String id);
    Map<String, Object> getDocument(String index, String id);
    <T> T getObject(String index, String id, Class<T> clazz);
    List<Map<String, Object>> searchByKeyword(String index, String field, String keyword, int size);
    boolean indexExists(String index);
    void createIndex(String index);
    void deleteIndex(String index);
    long countDocuments(String index);
    void bulkIndex(String index, List<Map<String, Object>> documents);
}
