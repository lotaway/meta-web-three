package com.metawebthree.mes.domain;

public class QcConstants {
    
    public static final String DEFAULT_AQL = "0.65";
    public static final String DEFAULT_INSPECTION_LEVEL = "normal";
    public static final int DEFAULT_SAMPLE_SIZE = 0;
    public static final String DEFAULT_ACCEPT_NUMBER = "0";
    public static final String DEFAULT_REJECT_NUMBER = "1";
    public static final String DEFAULT_DISPOSITION_RULE = "default";
    public static final String DEFAULT_QUALIFIED_FLOW = "pass";
    public static final String DEFAULT_UNQUALIFIED_FLOW = "isolate";
    public static final int DEFAULT_SORT_ORDER = 0;
    
    public static final int DEFAULT_SEVERITY = 1;
    public static final boolean DEFAULT_IS_MANDATORY = true;
    
    public static final int SCRAP_TIMEOUT_IDENTIFY = 1;
    public static final int SCRAP_TIMEOUT_ISOLATE = 2;
    public static final int SCRAP_TIMEOUT_EVALUATE = 24;
    public static final int SCRAP_TIMEOUT_DECIDE = 8;
    public static final int SCRAP_TIMEOUT_EXECUTE = 24;
    public static final int SCRAP_TIMEOUT_VERIFY = 4;
    public static final int SCRAP_TIMEOUT_CLOSE = 1;
    
    public static final int REWORK_TIMEOUT_IDENTIFY = 1;
    public static final int REWORK_TIMEOUT_ISOLATE = 2;
    public static final int REWORK_TIMEOUT_EVALUATE = 24;
    public static final int REWORK_TIMEOUT_DECIDE = 8;
    public static final int REWORK_TIMEOUT_EXECUTE = 48;
    public static final int REWORK_TIMEOUT_VERIFY = 8;
    public static final int REWORK_TIMEOUT_CLOSE = 1;
    
    public static final int RETENTION_DAYS_PROCESS = 365;
    public static final int RETENTION_DAYS_QC_RESULT = 730;
    public static final int RETENTION_DAYS_OPERATOR = 365;
    public static final int RETENTION_DAYS_EQUIPMENT = 365;
    
    private QcConstants() {
    }
}