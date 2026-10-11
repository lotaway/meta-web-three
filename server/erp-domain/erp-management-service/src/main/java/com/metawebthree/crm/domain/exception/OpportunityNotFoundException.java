package com.metawebthree.crm.domain.exception;

public class OpportunityNotFoundException extends RuntimeException {
    public OpportunityNotFoundException(Long id) {
        super("Opportunity not found: " + id);
    }
}
