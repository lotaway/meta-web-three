package com.metawebthree.cs.domain.model;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class Faq {
    private Long id;
    private String question;
    private String answer;
    private String category;
    private List<String> keywords;
    private Integer hitCount;
    private Double relevanceScore;
    private Boolean enabled;
    private Integer priority;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public Faq() {}

    public Faq(String question, String answer, String category, List<String> keywords) {
        this.question = question;
        this.answer = answer;
        this.category = category;
        this.keywords = keywords;
        this.enabled = true;
        this.hitCount = 0;
        this.relevanceScore = 0.0;
        this.priority = 0;
        this.createTime = LocalDateTime.now();
        this.updateTime = LocalDateTime.now();
    }

    public boolean matches(String query) {
        if (query == null || query.isEmpty()) {
            return false;
        }
        String lowerQuery = query.toLowerCase();
        if (question.toLowerCase().contains(lowerQuery)) {
            return true;
        }
        if (keywords != null) {
            for (String keyword : keywords) {
                if (keyword.toLowerCase().contains(lowerQuery)) {
                    return true;
                }
            }
        }
        return false;
    }

    public void incrementHitCount() {
        this.hitCount = (this.hitCount == null ? 0 : this.hitCount) + 1;
    }

    public void updateRelevance(Double score) {
        this.relevanceScore = score;
    }
}