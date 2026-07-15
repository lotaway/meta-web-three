package com.metawebthree.crm.domain.exception;

public class LeadNotFoundException extends RuntimeException {
    public LeadNotFoundException(Long id) {
        super("Lead not found: " + id);
    }
}
