package com.metawebthree.common.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.bulk.BulkOperation;
import co.elastic.clients.elasticsearch._types.FieldValue;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class ElasticsearchService implements ElasticsearchOperations {

    private final ElasticsearchClient client;

    @Autowired
    public ElasticsearchService(ElasticsearchClient client) {
        this.client = client;
    }

    @Override
    public void indexDocument(String index, String id, Map<String, Object> document) {
        try {
            client.index(i -> i.index(index).id(id).document(document));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to index document", e);
        }
    }

    @Override
    public <T> void indexObject(String index, String id, T document) {
        try {
            client.index(i -> i.index(index).id(id).document(document));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to index object", e);
        }
    }

    @Override
    public void deleteDocument(String index, String id) {
        try {
            client.delete(d -> d.index(index).id(id));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to delete document", e);
        }
    }

    @Override
    public Map<String, Object> getDocument(String index, String id) {
        try {
            var response = client.get(g -> g.index(index).id(id), Map.class);
            return response.found() ? response.source() : null;
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to get document", e);
        }
    }

    @Override
    public <T> T getObject(String index, String id, Class<T> clazz) {
        try {
            var response = client.get(g -> g.index(index).id(id), clazz);
            return response.found() ? response.source() : null;
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to get object", e);
        }
    }

    @Override
    public List<Map<String, Object>> searchByKeyword(String index, String field, String keyword, int size) {
        try {
            SearchResponse<Map> response = client.search(s -> s
                .index(index)
                .query(q -> q.match(m -> m
                    .field(field)
                    .query(FieldValue.of(keyword))
                ))
                .size(size),
                Map.class
            );
            return extractResults(response);
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to search documents", e);
        }
    }

    private List<Map<String, Object>> extractResults(SearchResponse<Map> response) {
        List<Map<String, Object>> results = new ArrayList<>();
        for (var hit : response.hits().hits()) {
            Map<String, Object> source = hit.source();
            if (source != null) {
                source.put("id", hit.id());
                results.add(source);
            }
        }
        return results;
    }

    @Override
    public boolean indexExists(String index) {
        try {
            var response = client.indices().exists(e -> e.index(index));
            return response.value();
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to check index existence", e);
        }
    }

    @Override
    public void createIndex(String index) {
        try {
            client.indices().create(c -> c.index(index));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to create index", e);
        }
    }

    @Override
    public void deleteIndex(String index) {
        try {
            client.indices().delete(d -> d.index(index));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to delete index", e);
        }
    }

    @Override
    public long countDocuments(String index) {
        try {
            var response = client.count(c -> c.index(index));
            return response.count();
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to count documents", e);
        }
    }

    @Override
    public void bulkIndex(String index, List<Map<String, Object>> documents) {
        try {
            var bulkBuilder = new BulkOperation.Builder();
            for (Map<String, Object> doc : documents) {
                String id = String.valueOf(doc.getOrDefault("id", ""));
                bulkBuilder.index(idx -> idx
                    .index(index)
                    .id(id)
                    .document(doc)
                );
            }
            client.bulk(b -> b.operations(bulkBuilder.build()));
        } catch (Exception e) {
            throw new ElasticsearchException("Failed to bulk index documents", e);
        }
    }
}
