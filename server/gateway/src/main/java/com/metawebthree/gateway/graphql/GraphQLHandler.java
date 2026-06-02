package com.metawebthree.gateway.graphql;

import graphql.ExecutionInput;
import graphql.GraphQL;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class GraphQLHandler {

    private final GraphQL graphQL;

    public GraphQLHandler(GraphQL graphQL) {
        this.graphQL = graphQL;
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

                    ExecutionInput.Builder executionInputBuilder = ExecutionInput.newExecutionInput()
                            .query(query);

                    if (variables != null) {
                        executionInputBuilder.variables(variables);
                    }

                    if (operationName != null && !operationName.isEmpty()) {
                        executionInputBuilder.operationName(operationName);
                    }

                    return Mono.fromFuture(() -> graphQL.executeAsync(executionInputBuilder.build()))
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
                });
    }
}