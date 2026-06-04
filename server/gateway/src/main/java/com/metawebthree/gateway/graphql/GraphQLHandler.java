package com.metawebthree.gateway.graphql;

import graphql.ExecutionInput;
import graphql.GraphQL;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class GraphQLHandler {

    private static final Logger log = LoggerFactory.getLogger(GraphQLHandler.class);

    private final GraphQL graphQL;
    private final FederationRouter router;

    public GraphQLHandler(GraphQL graphQL, FederationRouter router) {
        this.graphQL = graphQL;
        this.router = router;
    }

    public Mono<ServerResponse> handleGraphQL(ServerRequest request) {
        return request.bodyToMono(Map.class)
                .flatMap(body -> {
                    String query = (String) body.get("query");
                    Map<String, Object> variables = (Map<String, Object>) body.get("variables");
                    String operationName = (String) body.get("operationName");

                    if (query == null || query.isEmpty()) {
                        return ServerResponse.badRequest()
                                .bodyValue(Map.of("errors", Map.of("message", "Query is required")));
                    }

                    return routeQuery(query, variables, operationName);
                });
    }

    private Mono<ServerResponse> routeQuery(String query, Map<String, Object> variables, String operationName) {
        String firstField = FederationRouter.extractFirstRootField(query);

        if (firstField == null) {
            return executeLocal(query, variables, operationName);
        }

        if ("_service".equals(firstField) || "_entities".equals(firstField)) {
            return executeLocal(query, variables, operationName);
        }

        String subgraphUrl = router.resolveSubgraphUrl(query);
        if (subgraphUrl != null) {
            log.debug("Proxying query to subgraph {}: {}", subgraphUrl, query.substring(0, Math.min(100, query.length())));
            try {
                Map<String, Object> body = new ConcurrentHashMap<>();
                body.put("query", query);
                if (variables != null && !variables.isEmpty()) {
                    body.put("variables", variables);
                }
                if (operationName != null && !operationName.isEmpty()) {
                    body.put("operationName", operationName);
                }
                Map<String, Object> subgraphResult = router.executeOnSubgraph(subgraphUrl, query, variables);
                if (subgraphResult != null) {
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(subgraphResult);
                }
            } catch (Exception e) {
                log.error("Subgraph proxy failed: {}", e.getMessage());
            }
        }

        log.warn("Falling back to local execution for query: {}", query);
        return executeLocal(query, variables, operationName);
    }

    private Mono<ServerResponse> executeLocal(String query, Map<String, Object> variables, String operationName) {
        ExecutionInput.Builder builder = ExecutionInput.newExecutionInput().query(query);
        if (variables != null) builder.variables(variables);
        if (operationName != null && !operationName.isEmpty()) builder.operationName(operationName);

        return Mono.fromFuture(() -> graphQL.executeAsync(builder.build()))
                .flatMap(executionResult -> {
                    Map<String, Object> result = new ConcurrentHashMap<>();

                    if (executionResult.getErrors() != null && !executionResult.getErrors().isEmpty()) {
                        result.put("errors", executionResult.getErrors().stream()
                                .map(error -> {
                                    Map<String, Object> errorMap = new ConcurrentHashMap<>();
                                    errorMap.put("message", error.getMessage());
                                    if (error.getLocations() != null) {
                                        errorMap.put("locations", error.getLocations());
                                    }
                                    return errorMap;
                                })
                                .toList());
                    }

                    if (executionResult.getData() != null) {
                        result.put("data", executionResult.getData());
                    }

                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(result);
                });
    }
}
