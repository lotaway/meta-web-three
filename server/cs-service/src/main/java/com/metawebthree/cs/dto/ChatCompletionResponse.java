package com.metawebthree.cs.dto;

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

    public String getId() { return id; }
    public String getObject() { return object; }
    public long getCreated() { return created; }
    public String getModel() { return model; }
    public Choice[] getChoices() { return choices; }

    public static class Choice {
        public int index;
        public Message message;

        public static class Message {
            public String role;
            public String content;
        }
    }
}
