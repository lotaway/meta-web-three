package com.metawebthree.promotion.interfaces.web.dto;

import java.util.List;

public class CouponBatchView {
    private String batchId;
    private List<String> codes;

    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }
    public List<String> getCodes() { return codes; }
    public void setCodes(List<String> codes) { this.codes = codes; }
}
