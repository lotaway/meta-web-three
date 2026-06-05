package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TraceRecord {
    private Long id;
    private String traceCode;
    private TraceType traceType;
    private String productCode;
    private String batchNo;
    private String sn;
    private String sourceTraceCode;
    private TraceSource source;
    private List<TraceRelation> relations;
    private LocalDateTime createdAt;

    public enum TraceType {
        PRODUCT, BATCH, MATERIAL, SN, WORK_ORDER, PROCESS, QC, EQUIPMENT, OPERATOR
    }

    public enum TraceSource {
        WORK_ORDER, PRODUCTION_TASK, MATERIAL_ISSUE, QC_INSPECTION, EQUIPMENT, ANDON
    }

    public static class TraceRelation {
        private String relatedCode;
        private TraceType relatedType;
        private String relationType;
        private Integer quantity;

        public String getRelatedCode() { return relatedCode; }
        public void setRelatedCode(String relatedCode) { this.relatedCode = relatedCode; }
        public TraceType getRelatedType() { return relatedType; }
        public void setRelatedType(TraceType relatedType) { this.relatedType = relatedType; }
        public String getRelationType() { return relationType; }
        public void setRelationType(String relationType) { this.relationType = relationType; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
    }

    public void create(String traceCode, TraceType traceType, String productCode) {
        this.traceCode = traceCode;
        this.traceType = traceType;
        this.productCode = productCode;
        this.relations = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
    }

    public void linkMaterial(String materialCode, String batchNo, Integer quantity) {
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(materialCode);
        relation.setRelatedType(TraceType.MATERIAL);
        relation.setRelationType("CONSUMED");
        relation.setQuantity(quantity);
        this.relations.add(relation);
    }

    public void linkChildBatch(String childBatchNo, Integer quantity) {
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(childBatchNo);
        relation.setRelatedType(TraceType.BATCH);
        relation.setRelationType("SPLIT_FROM");
        relation.setQuantity(quantity);
        this.relations.add(relation);
    }

    public void linkParentBatch(String parentBatchNo) {
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(parentBatchNo);
        relation.setRelatedType(TraceType.BATCH);
        relation.setRelationType("PARENT");
        this.relations.add(relation);
    }

    public void linkSn(String sn) {
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(sn);
        relation.setRelatedType(TraceType.SN);
        relation.setRelationType("BIND");
        this.relations.add(relation);
    }

    public Optional<TraceRelation> findRelation(String relatedCode) {
        if (relations == null) {
            return Optional.empty();
        }
        return relations.stream()
            .filter(r -> relatedCode.equals(r.getRelatedCode()))
            .findFirst();
    }

    public List<TraceRecord> forwardTrace() {
        List<TraceRecord> results = new ArrayList<>();
        if (relations == null) {
            return results;
        }
        for (TraceRelation relation : relations) {
            if ("SPLIT_FROM".equals(relation.getRelationType()) || 
                "CONSUMED".equals(relation.getRelationType())) {
                TraceRecord record = new TraceRecord();
                record.create(relation.getRelatedCode(), relation.getRelatedType(), null);
                results.add(record);
            }
        }
        return results;
    }

    public List<TraceRecord> backwardTrace() {
        List<TraceRecord> results = new ArrayList<>();
        if (relations == null) {
            return results;
        }
        for (TraceRelation relation : relations) {
            if ("PARENT".equals(relation.getRelationType())) {
                TraceRecord record = new TraceRecord();
                record.create(relation.getRelatedCode(), relation.getRelatedType(), null);
                results.add(record);
            }
        }
        return results;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTraceCode() { return traceCode; }
    public void setTraceCode(String traceCode) { this.traceCode = traceCode; }
    public TraceType getTraceType() { return traceType; }
    public void setTraceType(TraceType traceType) { this.traceType = traceType; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getBatchNo() { return batchNo; }
    public void setBatchNo(String batchNo) { this.batchNo = batchNo; }
    public String getSn() { return sn; }
    public void setSn(String sn) { this.sn = sn; }
    public String getSourceTraceCode() { return sourceTraceCode; }
    public void setSourceTraceCode(String sourceTraceCode) { this.sourceTraceCode = sourceTraceCode; }
    public TraceSource getSource() { return source; }
    public void setSource(TraceSource source) { this.source = source; }
    public List<TraceRelation> getRelations() { return relations; }
    public void setRelations(List<TraceRelation> relations) { this.relations = relations; }
    public LocalDateTime getCreatedAt() { return createdAt; }

    public static class TraceChain {
        private TraceRecord root;
        private List<TraceRecord> forwardPath;
        private List<TraceRecord> backwardPath;

        public TraceRecord getRoot() { return root; }
        public void setRoot(TraceRecord root) { this.root = root; }
        public List<TraceRecord> getForwardPath() { return forwardPath; }
        public void setForwardPath(List<TraceRecord> forwardPath) { this.forwardPath = forwardPath; }
        public List<TraceRecord> getBackwardPath() { return backwardPath; }
        public void setBackwardPath(List<TraceRecord> backwardPath) { this.backwardPath = backwardPath; }
    }
}