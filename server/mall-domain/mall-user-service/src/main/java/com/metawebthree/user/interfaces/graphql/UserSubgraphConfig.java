package com.metawebthree.user.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.user.application.UserService;
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
public class UserSubgraphConfig {

    private final UserService userService;
    private final ResourcePatternResolver resourceResolver;

    public UserSubgraphConfig(UserService userService, ResourcePatternResolver resourceResolver) {
        this.userService = userService;
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
                        .dataFetcher("user", this::userDataFetcher)
                        .dataFetcher("users", this::usersDataFetcher)
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
        Map<String, Object> rep = representations.get(0);
        String typeName = (String) rep.get("__typename");
        if ("User".equals(typeName)) {
            Object idObj = rep.get("id");
            Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf((String) idObj);
            return userService.getUserById(id);
        }
        return null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String tn = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(tn);
        }
        return null;
    }

    private Object userDataFetcher(DataFetchingEnvironment env) {
        Object idObj = env.getArgument("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf((String) idObj);
        return userService.getUserById(id);
    }

    private Map<String, Object> usersDataFetcher(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        if (page == null) page = 0;
        if (size == null) size = 10;
        var userPage = userService.getUserList(page, size);
        List<Map<String, Object>> edges = userPage.getRecords().stream().map(u -> {
            Map<String, Object> edge = new ConcurrentHashMap<>();
            edge.put("node", u);
            return edge;
        }).toList();
        Map<String, Object> pageInfo = new ConcurrentHashMap<>();
        pageInfo.put("hasNextPage", userPage.getPages() > page + 1);
        pageInfo.put("hasPreviousPage", page > 0);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("edges", edges);
        result.put("pageInfo", pageInfo);
        result.put("totalCount", userPage.getTotal());
        return result;
    }
}
