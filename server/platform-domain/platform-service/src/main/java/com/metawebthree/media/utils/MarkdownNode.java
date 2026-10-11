package com.metawebthree.media.utils;

import lombok.Data;

@Data
public class MarkdownNode {
    private String type;
    private String text;
    private String url;
    private String language;

    public MarkdownNode(String type, String text) {
        this.type = type;
        this.text = text;
    }

    public MarkdownNode(String type, String text, String url) {
        this.type = type;
        this.text = text;
        this.url = url;
    }

    public MarkdownNode(String type, String text, String language, boolean isCode) {
        this.type = type;
        this.text = text;
        this.language = language;
    }

    @Override
    public String toString() {
        if ("LINK".equals(type)) {
            return String.format("LINK: %s -> %s", text, url);
        } else if ("CODE_BLOCK".equals(type)) {
            return String.format("CODE_BLOCK (%s): %s", language,
                    text.length() > 50 ? text.substring(0, 47) + "..." : text);
        } else {
            return String.format("%s: %s", type, text);
        }
    }
}