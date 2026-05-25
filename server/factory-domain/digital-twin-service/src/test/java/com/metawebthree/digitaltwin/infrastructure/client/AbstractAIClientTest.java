package com.metawebthree.digitaltwin.infrastructure.client;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AbstractAIClientTest {

    @Test
    void AIClientRequest_shouldSerializeToJson() {
        AbstractAIClient.AIClientRequest request = new AbstractAIClient.AIClientRequest(
            "TEST_CAPABILITY",
            Map.of("key1", "value1", "key2", 123)
        );

        String json = request.toJson();
        assertTrue(json.contains("key1"));
        assertTrue(json.contains("value1"));
        assertTrue(json.contains("key2"));
    }

    @Test
    void AIClientResponse_success_shouldReturnSuccessState() {
        AbstractAIClient.AIClientResponse response = 
            AbstractAIClient.AIClientResponse.success("test data", 200);

        assertTrue(response.isSuccess());
        assertEquals("test data", response.getData());
        assertEquals(200, response.getStatusCode());
        assertNull(response.getError());
    }

    @Test
    void AIClientResponse_failure_shouldReturnFailureState() {
        AbstractAIClient.AIClientResponse response = 
            AbstractAIClient.AIClientResponse.failure("error message", 500);

        assertFalse(response.isSuccess());
        assertNull(response.getData());
        assertEquals("error message", response.getError());
        assertEquals(500, response.getStatusCode());
    }

    @Test
    void AIClientResponse_getDataAsMap_shouldParseJson() {
        AbstractAIClient.AIClientResponse response = 
            AbstractAIClient.AIClientResponse.success("{\"key\":\"value\"}", 200);

        Map<String, Object> data = response.getDataAsMap();
        assertEquals("value", data.get("key"));
    }

    @Test
    void AIClientResponse_getDataAsMap_shouldReturnEmptyMap_whenDataIsNull() {
        AbstractAIClient.AIClientResponse response = 
            AbstractAIClient.AIClientResponse.failure("error", 500);

        Map<String, Object> data = response.getDataAsMap();
        assertTrue(data.isEmpty());
    }
}