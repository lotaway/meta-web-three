package com.metawebthree.finance.application.event;

public interface WorkOrderCompletionProcessor {
    void onWorkOrderCompleted(String message);
    void onTaskCompleted(String message);
}
