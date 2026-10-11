package com.metawebthree.project.domain.exception;

public class TimeEntryNotFoundException extends RuntimeException {
    public TimeEntryNotFoundException(String message) {
        super(message);
    }
}