package com.metawebthree.crm.interfaces.graphql.dto;

import com.metawebthree.crm.domain.entity.Opportunity;
import lombok.Data;

@Data
public class OpportunityDTO implements TypedNode {
    String __typename;
    String id;
    String opportunityNo;
    String title;
    String leadId;
    String customerId;
    String stage;
    Double amount;
    Integer probability;
    String expectedCloseDate;
    String actualCloseDate;
    String competitor;
    String description;
    String assignedTo;
    String createdAt;
    String updatedAt;

    @Override
    public String __typename() { return __typename; }

    public static OpportunityDTO from(Opportunity opp) {
        OpportunityDTO dto = new OpportunityDTO();
        dto.__typename = "Opportunity";
        baseFields(dto, opp);
        relationFields(dto, opp);
        pipelineFields(dto, opp);
        timestampFields(dto, opp);
        return dto;
    }

    private static void baseFields(OpportunityDTO dto, Opportunity opp) {
        dto.id = opp.getId().toString();
        dto.opportunityNo = opp.getOpportunityNo();
        dto.title = opp.getTitle();
    }

    private static void relationFields(OpportunityDTO dto, Opportunity opp) {
        dto.leadId = opp.getLeadId() != null ? opp.getLeadId().toString() : null;
        dto.customerId = opp.getCustomerId() != null ? opp.getCustomerId().toString() : null;
    }

    private static void pipelineFields(OpportunityDTO dto, Opportunity opp) {
        dto.stage = opp.getStage();
        dto.amount = opp.getAmount() != null ? opp.getAmount().doubleValue() : null;
        dto.probability = opp.getProbability();
        dto.expectedCloseDate = str(opp.getExpectedCloseDate());
        dto.actualCloseDate = str(opp.getActualCloseDate());
        dto.competitor = opp.getCompetitor();
        dto.description = opp.getDescription();
        dto.assignedTo = opp.getAssignedTo();
    }

    private static void timestampFields(OpportunityDTO dto, Opportunity opp) {
        dto.createdAt = str(opp.getCreatedAt());
        dto.updatedAt = str(opp.getUpdatedAt());
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }
}
