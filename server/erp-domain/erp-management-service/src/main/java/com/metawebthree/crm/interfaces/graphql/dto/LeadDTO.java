package com.metawebthree.crm.interfaces.graphql.dto;

import com.metawebthree.crm.domain.entity.Lead;
import lombok.Data;

@Data
public class LeadDTO implements TypedNode {
    String __typename;
    String id;
    String leadNo;
    String name;
    String company;
    String email;
    String phone;
    String source;
    String status;
    Integer score;
    String industry;
    String city;
    String description;
    String assignedTo;
    String createdAt;
    String updatedAt;

    @Override
    public String __typename() { return __typename; }

    public static LeadDTO from(Lead lead) {
        LeadDTO dto = new LeadDTO();
        dto.__typename = "Lead";
        baseFields(dto, lead);
        contactFields(dto, lead);
        pipelineFields(dto, lead);
        timestampFields(dto, lead);
        return dto;
    }

    private static void baseFields(LeadDTO dto, Lead lead) {
        dto.id = lead.getId().toString();
        dto.leadNo = lead.getLeadNo();
        dto.name = lead.getName();
        dto.company = lead.getCompany();
        dto.email = lead.getEmail();
        dto.phone = lead.getPhone();
    }

    private static void contactFields(LeadDTO dto, Lead lead) {
        dto.source = lead.getSource();
        dto.status = lead.getStatus();
        dto.score = lead.getScore();
        dto.industry = lead.getIndustry();
        dto.city = lead.getCity();
    }

    private static void pipelineFields(LeadDTO dto, Lead lead) {
        dto.description = lead.getDescription();
        dto.assignedTo = lead.getAssignedTo();
    }

    private static void timestampFields(LeadDTO dto, Lead lead) {
        dto.createdAt = str(lead.getCreatedAt());
        dto.updatedAt = str(lead.getUpdatedAt());
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }
}
