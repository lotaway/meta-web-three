package com.metawebthree.product.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.product.application.ProductCategoryApplicationService;
import com.metawebthree.product.application.ProductService;
import graphql.GraphQL;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import graphql.schema.DataFetchingEnvironment;
import graphql.TypeResolutionEnvironment;
import graphql.schema.GraphQLObjectType;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Configuration
public class ProductSubgraphConfig {

    private final ProductService productService;
    private final ProductCategoryApplicationService categoryService;
    private final ResourcePatternResolver resourceResolver;

    public ProductSubgraphConfig(ProductService productService,
                                  ProductCategoryApplicationService categoryService,
                                  ResourcePatternResolver resourceResolver) {
        this.productService = productService;
        this.categoryService = categoryService;
        this.resourceResolver = resourceResolver;
    }

    @Bean
    public GraphQL graphQL() throws IOException {
        TypeDefinitionRegistry typeRegistry = buildSchema();
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = buildFederationSchema(typeRegistry, wiring);
        return GraphQL.newGraphQL(schema).build();
    }

    private TypeDefinitionRegistry buildSchema() throws IOException {
        Resource[] resources = resourceResolver.getResources("classpath:graphql/*.graphqls");
        StringBuilder sb = new StringBuilder();
        for (Resource r : resources) {
            sb.append(new String(r.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }
        return new SchemaParser().parse(sb.toString());
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("category", this::categoryDataFetcher)
                        .dataFetcher("categories", this::categoriesDataFetcher)
                        .dataFetcher("product", this::productDataFetcher)
                        .dataFetcher("products", this::productsDataFetcher)
                        .dataFetcher("productBySku", this::productBySkuDataFetcher)
                )
                .type("Mutation", wiring -> wiring
                        .dataFetcher("createProduct", this::createProductDataFetcher)
                        .dataFetcher("updateProduct", this::updateProductDataFetcher)
                        .dataFetcher("deleteProduct", this::deleteProductDataFetcher)
                )
                .build();
    }

    private GraphQLSchema buildFederationSchema(TypeDefinitionRegistry registry, RuntimeWiring wiring) {
        return Federation.transform(registry, wiring)
                .fetchEntities(this::fetchEntity)
                .resolveEntityType(this::resolveEntityType)
                .build();
    }

    private Object fetchEntity(DataFetchingEnvironment env) {
        List<Map<String, Object>> representations = env.getArgument("representations");
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        if ("Product".equals(typeName)) {
            Object idObj = representation.get("id");
            Integer id = idObj instanceof Integer ? (Integer) idObj : Integer.valueOf((String) idObj);
            return productService.getProductById(id);
        }
        if ("Category".equals(typeName)) {
            Object idObj = representation.get("id");
            Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf((String) idObj);
            var subs = categoryService.findSubCategories(id);
            if (subs.isEmpty()) {
                var roots = categoryService.findSubCategories(0L);
                for (var c : roots) {
                    if (c.getId().equals(id)) return c;
                }
            }
            return null;
        }
        return null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String typeName = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(typeName);
        }
        return null;
    }

    private Map<String, Object> categoryDataFetcher(DataFetchingEnvironment env) {
        Object idObj = env.getArgument("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf((String) idObj);
        var subs = categoryService.findSubCategories(id);
        if (!subs.isEmpty()) {
            Map<String, Object> m = new ConcurrentHashMap<>();
            m.put("id", String.valueOf(id));
            m.put("name", subs.get(0).getName());
            m.put("parentId", subs.get(0).getParentId() != null ? String.valueOf(subs.get(0).getParentId()) : null);
            return m;
        }
        return null;
    }

    private Map<String, Object> categoriesDataFetcher(DataFetchingEnvironment env) {
        Object parentIdObj = env.getArgument("parentId");
        Long parentId = parentIdObj != null
            ? (parentIdObj instanceof Number ? ((Number) parentIdObj).longValue() : Long.valueOf((String) parentIdObj))
            : 0L;
        var cats = categoryService.findSubCategories(parentId);
        List<Map<String, Object>> edges = cats.stream().map(c -> {
            Map<String, Object> node = new ConcurrentHashMap<>();
            node.put("id", String.valueOf(c.getId()));
            node.put("name", c.getName());
            node.put("parentId", c.getParentId() != null ? String.valueOf(c.getParentId()) : null);
            node.put("level", c.getLevel());
            node.put("sortOrder", c.getSort());
            Map<String, Object> edge = new ConcurrentHashMap<>();
            edge.put("node", node);
            return edge;
        }).toList();
        Map<String, Object> pageInfo = new ConcurrentHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("edges", edges);
        result.put("pageInfo", pageInfo);
        result.put("totalCount", cats.size());
        return result;
    }

    private Object productDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return productService.getProductById(Integer.valueOf(id));
    }

    private Map<String, Object> productsDataFetcher(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        if (page == null) page = 0;
        if (size == null) size = 10;
        var products = productService.listProducts(null, null, null);
        List<Map<String, Object>> edges = products.stream().map(p -> {
            Map<String, Object> edge = new ConcurrentHashMap<>();
            edge.put("node", p);
            return edge;
        }).toList();
        Map<String, Object> pageInfo = new ConcurrentHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("edges", edges);
        result.put("pageInfo", pageInfo);
        result.put("totalCount", products.size());
        return result;
    }

    private Object productBySkuDataFetcher(DataFetchingEnvironment env) {
        String sku = env.getArgument("sku");
        var products = productService.simpleSearch(sku, 1);
        return products.isEmpty() ? null : products.get(0);
    }

    private Map<String, Object> createProductDataFetcher(DataFetchingEnvironment env) {
        Map<String, Object> input = env.getArgument("input");
        productService.createProduct();
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("name", input.get("name"));
        result.put("sku", input.get("sku"));
        return result;
    }

    private Map<String, Object> updateProductDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        productService.updateProduct(Long.valueOf(id), null);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("id", id);
        return result;
    }

    private Boolean deleteProductDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        productService.deleteProduct(id);
        return true;
    }
}
