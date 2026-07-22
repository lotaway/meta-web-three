package com.metawebthree.crm.interfaces.graphql;

import graphql.ExecutionInput;
import graphql.GraphQL;
import graphql.ExecutionResult;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/graphql")
public class CrmSubgraphController {

    public record GraphQLRequest(String query, Map<String, Object> variables, String operationName) {
        static GraphQLRequest fromBody(Map<String, Object> body) {
            return new GraphQLRequest(
                (String) body.get("query"),
                (Map<String, Object>) body.get("variables"),
                (String) body.get("operationName")
            );
        }
    }

    public record GraphQLError(String message) {}

    public record GraphQLResponse(Map<String, Object> data, List<GraphQLError> errors) {
        static GraphQLResponse fromExecutionResult(ExecutionResult result) {
            List<GraphQLError> errors = result.getErrors() != null && !result.getErrors().isEmpty()
                ? result.getErrors().stream().map(e -> new GraphQLError(e.getMessage())).toList()
                : null;
            return new GraphQLResponse(result.getData(), errors);
        }
    }

    private final GraphQL graphQL;

    public CrmSubgraphController(GraphQL graphQL) {
        this.graphQL = graphQL;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public CompletableFuture<GraphQLResponse> execute(@RequestBody Map<String, Object> body) {
        GraphQLRequest request = GraphQLRequest.fromBody(body);
        ExecutionInput input = ExecutionInput.newExecutionInput()
            .query(request.query())
            .variables(request.variables() != null ? request.variables() : new ConcurrentHashMap<>())
            .operationName(request.operationName())
            .build();
        return graphQL.executeAsync(input).thenApply(GraphQLResponse::fromExecutionResult);
    }
}
