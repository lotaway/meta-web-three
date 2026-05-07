package com.metawebthree.product.application;

import com.metawebthree.common.elasticsearch.ElasticsearchOperations;
import com.metawebthree.product.domain.model.ProductDO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class EsProductApplicationService {

    private static final String INDEX_NAME = "product";

    private final ElasticsearchOperations esOperations;
    private final ProductMapper productMapper;

    public EsProductApplicationService(
            @Qualifier("elasticsearchService") ElasticsearchOperations esOperations,
            ProductMapper productMapper) {
        this.esOperations = esOperations;
        this.productMapper = productMapper;
    }

    public int importAllToEs() {
        if (!esOperations.indexExists(INDEX_NAME)) {
            esOperations.createIndex(INDEX_NAME);
        }

        List<ProductDO> products = productMapper.selectList(null);
        List<Map<String, Object>> docs = buildDocuments(products);
        esOperations.bulkIndex(INDEX_NAME, docs);
        return docs.size();
    }

    public void syncToEs(Integer id) {
        ProductDO product = productMapper.selectById(id);
        if (product == null) {
            return;
        }

        Map<String, Object> doc = buildDocument(product);
        esOperations.indexDocument(INDEX_NAME, String.valueOf(id), doc);
    }

    private List<Map<String, Object>> buildDocuments(List<ProductDO> products) {
        return products.stream()
                .map(this::buildDocument)
                .toList();
    }

    private Map<String, Object> buildDocument(ProductDO product) {
        Map<String, Object> doc = new HashMap<>();
        doc.put("id", product.getId());
        doc.put("productName", product.getProductName());
        doc.put("productNo", product.getProductNo());
        doc.put("productRemark", product.getProductRemark());
        doc.put("brandId", product.getBrandId());
        doc.put("categoryId", product.getCategoryId());
        doc.put("createTime", product.getCreateTime());
        return doc;
    }
}
