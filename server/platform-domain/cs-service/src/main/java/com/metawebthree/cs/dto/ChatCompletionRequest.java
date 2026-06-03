package com.metawebthree.cs.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ChatCompletionRequest {
    private String model;
    private List<Map<String, Object>> messages;
    private boolean stream;
    private List<Map<String, Object>> tools;
    private String toolChoice;

    public ChatCompletionRequest(String model, List<Map<String, Object>> messages) {
        this.model = model;
        this.messages = messages;
        this.stream = false;
    }
}
