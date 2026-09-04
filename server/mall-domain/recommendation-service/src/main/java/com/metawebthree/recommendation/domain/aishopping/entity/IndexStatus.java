package com.metawebthree.recommendation.domain.aishopping.entity;

import java.time.LocalDateTime;
import java.util.Map;

/** Index build task status. */
public class IndexStatus {

    public enum State {
        IDLE, RUNNING, COMPLETED, FAILED
    }

    private State state = State.IDLE;
    private String currentType = "all";
    private int total;
    private int processed;
    private int textVectorCount;
    private int imageVectorCount;
    private String lastError;
    private LocalDateTime startedAt;
    private LocalDateTime finishedAt;

    public State getState() {
        return state;
    }

    public void setState(State state) {
        this.state = state;
    }

    public String getCurrentType() {
        return currentType;
    }

    public void setCurrentType(String currentType) {
        this.currentType = currentType;
    }

    public int getTotal() {
        return total;
    }

    public void setTotal(int total) {
        this.total = total;
    }

    public int getProcessed() {
        return processed;
    }

    public void setProcessed(int processed) {
        this.processed = processed;
    }

    public int getTextVectorCount() {
        return textVectorCount;
    }

    public void setTextVectorCount(int textVectorCount) {
        this.textVectorCount = textVectorCount;
    }

    public int getImageVectorCount() {
        return imageVectorCount;
    }

    public void setImageVectorCount(int imageVectorCount) {
        this.imageVectorCount = imageVectorCount;
    }

    public String getLastError() {
        return lastError;
    }

    public void setLastError(String lastError) {
        this.lastError = lastError;
    }

    public LocalDateTime getStartedAt() {
        return startedAt;
    }

    public void setStartedAt(LocalDateTime startedAt) {
        this.startedAt = startedAt;
    }

    public LocalDateTime getFinishedAt() {
        return finishedAt;
    }

    public void setFinishedAt(LocalDateTime finishedAt) {
        this.finishedAt = finishedAt;
    }

    public double getProgress() {
        return total > 0 ? Math.min(1.0, (double) processed / total) : 0.0;
    }

    public Map<String, Object> toMap() {
        return Map.of(
                "state", state.name(),
                "currentType", currentType,
                "total", total,
                "processed", processed,
                "textVectorCount", textVectorCount,
                "imageVectorCount", imageVectorCount,
                "progress", getProgress(),
                "lastError", lastError != null ? lastError : "",
                "startedAt", startedAt != null ? startedAt.toString() : "",
                "finishedAt", finishedAt != null ? finishedAt.toString() : "");
    }
}
