package com.metawebthree.cs.dto;

import java.util.List;
import java.util.Map;

public class ChatCompletionRequest {
    private String model;
    private List<Map<String, String>> messages;
    private boolean stream;

    public ChatCompletionRequest(String model, List<Map<String, String>> messages) {
        this.model = model;
        this.messages = messages;
        this.stream = false;
    }

    public String getModel() { return model; }
    public List<Map<String, String>> getMessages() { return messages; }
    public boolean isStream() { return stream; }
}
