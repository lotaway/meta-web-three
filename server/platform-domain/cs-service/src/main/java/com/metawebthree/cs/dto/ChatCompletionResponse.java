package com.metawebthree.cs.dto;

import lombok.Data;

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

    @Data
    public static class Choice {
        public int index;
        public Message message;

        @Data
        public static class Message {
            public String role;
            public String content;
        }
    }
}
