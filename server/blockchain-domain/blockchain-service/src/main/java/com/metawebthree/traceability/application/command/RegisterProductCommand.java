package com.metawebthree.traceability.application.command;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class RegisterProductCommand {

    private String productId;

    private String productName;

    private String category;

    private String manufacturer;

    private String productionLocation;

    private LocalDateTime productionDate;
}