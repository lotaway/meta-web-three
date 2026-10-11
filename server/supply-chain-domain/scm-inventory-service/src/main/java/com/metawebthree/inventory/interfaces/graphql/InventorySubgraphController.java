package com.metawebthree.inventory.interfaces.graphql;

import graphql.ExecutionInput;
import graphql.GraphQL;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/graphql")
public class InventorySubgraphController {

    private final GraphQL graphQL;

    public InventorySubgraphController(GraphQL graphQL) {
        this.graphQL = graphQL;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public CompletableFuture<Map<String, Object>> execute(@RequestBody Map<String, Object> body) {
        String query = (String) body.get("query");
        Map<String, Object> variables = (Map<String, Object>) body.get("variables");
        String operationName = (String) body.get("operationName");

        ExecutionInput.Builder builder = ExecutionInput.newExecutionInput().query(query);
        if (variables != null) builder.variables(variables);
        if (operationName != null && !operationName.isEmpty()) builder.operationName(operationName);

        return graphQL.executeAsync(builder.build())
                .thenApply(result -> {
                    Map<String, Object> response = new java.util.concurrent.ConcurrentHashMap<>();
                    if (result.getErrors() != null && !result.getErrors().isEmpty()) {
                        response.put("errors", result.getErrors().stream().map(e -> {
                            Map<String, Object> err = new java.util.concurrent.ConcurrentHashMap<>();
                            err.put("message", e.getMessage());
                            return err;
                        }).toList());
                    }
                    if (result.getData() != null) {
                        response.put("data", result.getData());
                    }
                    return response;
                });
    }
}
