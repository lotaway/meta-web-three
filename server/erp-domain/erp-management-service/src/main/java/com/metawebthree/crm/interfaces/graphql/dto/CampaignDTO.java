package com.metawebthree.crm.interfaces.graphql.dto;

import com.metawebthree.crm.domain.entity.Campaign;
import lombok.Data;

@Data
public class CampaignDTO implements TypedNode {
    String __typename;
    String id;
    String name;
    String description;
    String type;
    String status;
    String startDate;
    String endDate;
    Double budget;
    Double actualCost;
    Double expectedRevenue;
    Integer leadsGenerated;
    Integer convertedCustomers;
    String createdAt;
    String updatedAt;

    @Override
    public String __typename() { return __typename; }

    public static CampaignDTO from(Campaign campaign) {
        CampaignDTO dto = new CampaignDTO();
        dto.__typename = "Campaign";
        baseFields(dto, campaign);
        dateFields(dto, campaign);
        financialFields(dto, campaign);
        timestampFields(dto, campaign);
        return dto;
    }

    private static void baseFields(CampaignDTO dto, Campaign campaign) {
        dto.id = campaign.getId().toString();
        dto.name = campaign.getName();
        dto.description = campaign.getDescription();
        dto.type = campaign.getType();
        dto.status = campaign.getStatus();
    }

    private static void dateFields(CampaignDTO dto, Campaign campaign) {
        dto.startDate = str(campaign.getStartDate());
        dto.endDate = str(campaign.getEndDate());
    }

    private static void financialFields(CampaignDTO dto, Campaign campaign) {
        dto.budget = campaign.getBudget() != null ? campaign.getBudget().doubleValue() : null;
        dto.actualCost = campaign.getActualCost() != null ? campaign.getActualCost().doubleValue() : null;
        dto.expectedRevenue = campaign.getExpectedRevenue() != null ? campaign.getExpectedRevenue().doubleValue() : null;
        dto.leadsGenerated = campaign.getLeadsGenerated();
        dto.convertedCustomers = campaign.getConvertedCustomers();
    }

    private static void timestampFields(CampaignDTO dto, Campaign campaign) {
        dto.createdAt = str(campaign.getCreatedAt());
        dto.updatedAt = str(campaign.getUpdatedAt());
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }
}
