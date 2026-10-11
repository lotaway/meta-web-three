package com.metawebthree.crm.domain.exception;

public class CampaignNotFoundException extends RuntimeException {
    public CampaignNotFoundException(Long id) {
        super("Campaign not found: " + id);
    }
}
