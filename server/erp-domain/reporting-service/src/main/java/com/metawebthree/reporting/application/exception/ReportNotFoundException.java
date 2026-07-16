package com.metawebthree.reporting.application.exception;

public class ReportNotFoundException extends RuntimeException {
    public ReportNotFoundException(String reportNo) {
        super("Report not found: " + reportNo);
    }
}
