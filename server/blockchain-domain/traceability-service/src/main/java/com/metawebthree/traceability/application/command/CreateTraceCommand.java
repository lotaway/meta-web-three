package com.metawebthree.traceability.application.command;

import lombok.Data;

@Data
public class CreateTraceCommand {

    private String productId;

    private String batchNumber;
}