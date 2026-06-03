package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class CategoryClient {

    @DubboReference
    private CategoryService categoryService;

    /**
     * Get category by ID
     * @param id category ID
     * @return category data map
     */
    public Map<String, Object> getCategoryById(String id) {
        try {
            GetCategoryByIdRequest request = GetCategoryByIdRequest.newBuilder()
                    .setId(Long.parseLong(id))
                    .build();
            GetCategoryByIdResponse response = categoryService.getCategoryById(request);
            
            Map<String, Object> result = new HashMap<>();
            CategoryDTO category = response.getCategory();
            if (category != null && category.getId() > 0) {
                result.put("id", category.getId());
                result.put("name", category.getName());
                result.put("parentId", category.getParentId());
                result.put("sortOrder", category.getSortOrder());
                result.put("icon", category.getIcon());
                result.put("enabled", category.getEnabled());
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get category by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get all categories
     * @return categories connection
     */
    public Map<String, Object> getCategories() {
        try {
            GetCategoriesRequest request = GetCategoriesRequest.newBuilder()
                    .setPage(0)
                    .setSize(100)
                    .build();
            GetCategoriesResponse response = categoryService.getCategories(request);
            
            Map<String, Object> connection = new HashMap<>();
            List<Map<String, Object>> edges = new ArrayList<>();
            
            for (CategoryDTO category : response.getCategoriesList()) {
                Map<String, Object> edge = new HashMap<>();
                Map<String, Object> node = new HashMap<>();
                node.put("id", category.getId());
                node.put("name", category.getName());
                node.put("parentId", category.getParentId());
                node.put("sortOrder", category.getSortOrder());
                node.put("icon", category.getIcon());
                node.put("enabled", category.getEnabled());
                edge.put("node", node);
                edges.add(edge);
            }
            
            connection.put("edges", edges);
            connection.put("totalCount", response.getTotalCount());
            connection.put("pageInfo", Map.of(
                "hasNextPage", false,
                "hasPreviousPage", false
            ));
            return connection;
        } catch (Exception e) {
            log.error("Failed to get categories: error: {}", e.getMessage());
        }
        return createEmptyCategoriesConnection();
    }

    /**
     * Get category tree
     * @return category tree
     */
    public List<Map<String, Object>> getCategoryTree() {
        try {
            GetCategoryTreeRequest request = GetCategoryTreeRequest.newBuilder().build();
            GetCategoryTreeResponse response = categoryService.getCategoryTree(request);
            
            List<Map<String, Object>> tree = new ArrayList<>();
            for (CategoryNodeDTO node : response.getTreeList()) {
                tree.add(convertNodeToMap(node));
            }
            return tree;
        } catch (Exception e) {
            log.error("Failed to get category tree: error: {}", e.getMessage());
        }
        return new ArrayList<>();
    }

    private Map<String, Object> convertNodeToMap(CategoryNodeDTO node) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", node.getId());
        map.put("name", node.getName());
        map.put("parentId", node.getParentId());
        map.put("sortOrder", node.getSortOrder());
        map.put("icon", node.getIcon());
        map.put("enabled", node.getEnabled());
        
        if (node.getChildrenList() != null && !node.getChildrenList().isEmpty()) {
            List<Map<String, Object>> children = new ArrayList<>();
            for (CategoryNodeDTO child : node.getChildrenList()) {
                children.add(convertNodeToMap(child));
            }
            map.put("children", children);
        }
        
        return map;
    }

    private Map<String, Object> createEmptyCategoriesConnection() {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", new ArrayList<>());
        connection.put("totalCount", 0);
        connection.put("pageInfo", Map.of(
            "hasNextPage", false,
            "hasPreviousPage", false
        ));
        return connection;
    }
}