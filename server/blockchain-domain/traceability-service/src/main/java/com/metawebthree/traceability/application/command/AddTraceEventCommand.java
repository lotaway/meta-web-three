package com.metawebthree.traceability.application.command;

import lombok.Data;

@Data
public class AddTraceEventCommand {

    private Long traceId;

    private String eventType;

    private String description;

    private String location;

    private String extraData;
}