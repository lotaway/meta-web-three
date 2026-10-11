package com.metawebthree.cs.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ChatCompletionResponse {
    private String id;
    private String object;
    private long created;
    private String model;
    private Choice[] choices;

    public String extractContent() {
        if (choices != null && choices.length > 0 && choices[0].message != null) {
            return choices[0].message.content;
        }
        return "";
    }

    public boolean hasToolCalls() {
        return choices != null && choices.length > 0
                && choices[0].message != null
                && choices[0].message.toolCalls != null
                && !choices[0].message.toolCalls.isEmpty();
    }

    public List<Map<String, Object>> getToolCalls() {
        if (!hasToolCalls()) return List.of();
        return choices[0].message.toolCalls;
    }

    public String getFinishReason() {
        if (choices != null && choices.length > 0) {
            return choices[0].finishReason;
        }
        return "";
    }

    @Data
    public static class Choice {
        public int index;
        public Message message;
        public String finishReason;

        @Data
        public static class Message {
            public String role;
            public String content;
            public List<Map<String, Object>> toolCalls;
        }
    }
}
