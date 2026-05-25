package com.metawebthree.digitaltwin.common;

public final class DigitalTwinPermissions {

    private DigitalTwinPermissions() {}

    public static final String DEVICE_READ    = "dt:device:read";
    public static final String DEVICE_CREATE  = "dt:device:create";
    public static final String DEVICE_UPDATE  = "dt:device:update";
    public static final String DEVICE_CONTROL = "dt:device:control";

    public static final String WORKSHOP_READ   = "dt:workshop:read";
    public static final String WORKSHOP_CREATE = "dt:workshop:create";
    public static final String WORKSHOP_UPDATE = "dt:workshop:update";

    public static final String PRODUCTION_LINE_READ   = "dt:production-line:read";
    public static final String PRODUCTION_LINE_CREATE = "dt:production-line:create";
    public static final String PRODUCTION_LINE_UPDATE = "dt:production-line:update";

    public static final String ALERT_READ    = "dt:alert:read";
    public static final String ALERT_CREATE  = "dt:alert:create";
    public static final String ALERT_ACK     = "dt:alert:ack";
    public static final String ALERT_RESOLVE = "dt:alert:resolve";

    public static final String STATS_READ = "dt:stats:read";
}
