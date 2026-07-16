package com.metawebthree.crm.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.crm.application.query.LeadQueryService;
import com.metawebthree.crm.application.query.OpportunityQueryService;
import com.metawebthree.crm.application.query.TicketQueryService;
import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.entity.Contact;
import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import com.metawebthree.crm.domain.repository.ContactRepository;
import graphql.GraphQL;
import graphql.TypeResolutionEnvironment;
import graphql.schema.DataFetchingEnvironment;
import graphql.schema.GraphQLObjectType;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Configuration
public class CrmSubgraphConfig {

    private final LeadQueryService leadQueryService;
    private final OpportunityQueryService opportunityQueryService;
    private final TicketQueryService ticketQueryService;
    private final CampaignRepository campaignRepository;
    private final ContactRepository contactRepository;
    private final ResourcePatternResolver resourceResolver;

    public CrmSubgraphConfig(LeadQueryService leadQueryService,
                             OpportunityQueryService opportunityQueryService,
                             TicketQueryService ticketQueryService,
                             CampaignRepository campaignRepository,
                             ContactRepository contactRepository,
                             ResourcePatternResolver resourceResolver) {
        this.leadQueryService = leadQueryService;
        this.opportunityQueryService = opportunityQueryService;
        this.ticketQueryService = ticketQueryService;
        this.campaignRepository = campaignRepository;
        this.contactRepository = contactRepository;
        this.resourceResolver = resourceResolver;
    }

    @Bean
    public GraphQL graphQL() throws IOException {
        TypeDefinitionRegistry typeRegistry = buildSchema();
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = buildFederationSchema(typeRegistry, wiring);
        return GraphQL.newGraphQL(schema).build();
    }

    private TypeDefinitionRegistry buildSchema() throws IOException {
        Resource[] resources = resourceResolver.getResources("classpath:graphql/*.graphqls");
        StringBuilder sb = new StringBuilder();
        for (Resource r : resources) {
            sb.append(new String(r.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }
        return new SchemaParser().parse(sb.toString());
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("lead", this::leadDataFetcher)
                        .dataFetcher("leads", this::leadsDataFetcher)
                        .dataFetcher("opportunity", this::opportunityDataFetcher)
                        .dataFetcher("opportunities", this::opportunitiesDataFetcher)
                        .dataFetcher("ticket", this::ticketDataFetcher)
                        .dataFetcher("tickets", this::ticketsDataFetcher)
                        .dataFetcher("campaign", this::campaignDataFetcher)
                        .dataFetcher("campaigns", this::campaignsDataFetcher)
                        .dataFetcher("contact", this::contactDataFetcher)
                        .dataFetcher("contacts", this::contactsDataFetcher)
                )
                .build();
    }

    private GraphQLSchema buildFederationSchema(TypeDefinitionRegistry registry, RuntimeWiring wiring) {
        return Federation.transform(registry, wiring)
                .fetchEntities(this::fetchEntity)
                .resolveEntityType(this::resolveEntityType)
                .build();
    }

    @SuppressWarnings("unchecked")
    private Object fetchEntity(DataFetchingEnvironment env) {
        List<Map<String, Object>> representations = env.getArgument("representations");
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        Object idObj = representation.get("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf(idObj.toString());

        return switch (typeName) {
            case "Lead" -> {
                Lead lead = leadQueryService.getById(id);
                yield lead != null ? leadToMap(lead) : null;
            }
            case "Opportunity" -> {
                Opportunity opp = opportunityQueryService.getById(id);
                yield opp != null ? opportunityToMap(opp) : null;
            }
            case "Ticket" -> {
                CustomerServiceTicket ticket = ticketQueryService.getById(id);
                yield ticket != null ? ticketToMap(ticket) : null;
            }
            case "Campaign" -> {
                Campaign campaign = campaignRepository.selectById(id);
                yield campaign != null ? campaignToMap(campaign) : null;
            }
            case "Contact" -> {
                Contact contact = contactRepository.selectById(id);
                yield contact != null ? contactToMap(contact) : null;
            }
            default -> null;
        };
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String typeName = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(typeName);
        }
        return null;
    }

    private Map<String, Object> leadDataFetcher(DataFetchingEnvironment env) {
        Long id = Long.valueOf(env.getArgument("id"));
        Lead lead = leadQueryService.getById(id);
        return lead != null ? leadToMap(lead) : null;
    }

    private Map<String, Object> leadsDataFetcher(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        String source = env.getArgument("source");
        String keyword = env.getArgument("keyword");
        List<Lead> leads;
        if (keyword != null && !keyword.isBlank()) {
            leads = leadQueryService.search(keyword);
        } else if (status != null && !status.isBlank()) {
            leads = leadQueryService.listByStatus(status);
        } else if (source != null && !source.isBlank()) {
            leads = leadQueryService.listBySource(source);
        } else {
            leads = leadQueryService.listAll();
        }
        List<Map<String, Object>> edges = leads.stream().map(lead -> {
            Map<String, Object> edge = new LinkedHashMap<>();
            edge.put("node", leadToMap(lead));
            edge.put("cursor", lead.getId().toString());
            return edge;
        }).toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("edges", edges);
        result.put("totalCount", leads.size());
        Map<String, Object> pageInfo = new LinkedHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        result.put("pageInfo", pageInfo);
        return result;
    }

    private Map<String, Object> opportunityDataFetcher(DataFetchingEnvironment env) {
        Long id = Long.valueOf(env.getArgument("id"));
        Opportunity opp = opportunityQueryService.getById(id);
        return opp != null ? opportunityToMap(opp) : null;
    }

    private Map<String, Object> opportunitiesDataFetcher(DataFetchingEnvironment env) {
        String stage = env.getArgument("stage");
        String keyword = env.getArgument("keyword");
        List<Opportunity> opportunities;
        if (keyword != null && !keyword.isBlank()) {
            opportunities = opportunityQueryService.search(keyword);
        } else if (stage != null && !stage.isBlank()) {
            opportunities = opportunityQueryService.listByStage(stage);
        } else {
            opportunities = opportunityQueryService.listAll();
        }
        List<Map<String, Object>> edges = opportunities.stream().map(opp -> {
            Map<String, Object> edge = new LinkedHashMap<>();
            edge.put("node", opportunityToMap(opp));
            edge.put("cursor", opp.getId().toString());
            return edge;
        }).toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("edges", edges);
        result.put("totalCount", opportunities.size());
        Map<String, Object> pageInfo = new LinkedHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        result.put("pageInfo", pageInfo);
        return result;
    }

    private Map<String, Object> ticketDataFetcher(DataFetchingEnvironment env) {
        Long id = Long.valueOf(env.getArgument("id"));
        CustomerServiceTicket ticket = ticketQueryService.getById(id);
        return ticket != null ? ticketToMap(ticket) : null;
    }

    private Map<String, Object> ticketsDataFetcher(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        String priority = env.getArgument("priority");
        List<CustomerServiceTicket> tickets;
        if (priority != null && !priority.isBlank()) {
            tickets = ticketQueryService.listByPriority(priority);
        } else if (status != null && !status.isBlank()) {
            tickets = ticketQueryService.listByStatus(status);
        } else {
            tickets = ticketQueryService.listAll();
        }
        List<Map<String, Object>> edges = tickets.stream().map(ticket -> {
            Map<String, Object> edge = new LinkedHashMap<>();
            edge.put("node", ticketToMap(ticket));
            edge.put("cursor", ticket.getId().toString());
            return edge;
        }).toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("edges", edges);
        result.put("totalCount", tickets.size());
        Map<String, Object> pageInfo = new LinkedHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        result.put("pageInfo", pageInfo);
        return result;
    }

    private Map<String, Object> campaignDataFetcher(DataFetchingEnvironment env) {
        Long id = Long.valueOf(env.getArgument("id"));
        Campaign campaign = campaignRepository.selectById(id);
        return campaign != null ? campaignToMap(campaign) : null;
    }

    private Map<String, Object> campaignsDataFetcher(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        String type = env.getArgument("type");
        List<Campaign> campaigns = campaignRepository.selectList(null);
        if (status != null && !status.isBlank()) {
            campaigns = campaigns.stream().filter(c -> status.equals(c.getStatus())).toList();
        }
        if (type != null && !type.isBlank()) {
            campaigns = campaigns.stream().filter(c -> type.equals(c.getType())).toList();
        }
        List<Map<String, Object>> edges = campaigns.stream().map(campaign -> {
            Map<String, Object> edge = new LinkedHashMap<>();
            edge.put("node", campaignToMap(campaign));
            edge.put("cursor", campaign.getId().toString());
            return edge;
        }).toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("edges", edges);
        result.put("totalCount", campaigns.size());
        Map<String, Object> pageInfo = new LinkedHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        result.put("pageInfo", pageInfo);
        return result;
    }

    private Map<String, Object> contactDataFetcher(DataFetchingEnvironment env) {
        Long id = Long.valueOf(env.getArgument("id"));
        Contact contact = contactRepository.selectById(id);
        return contact != null ? contactToMap(contact) : null;
    }

    private Map<String, Object> contactsDataFetcher(DataFetchingEnvironment env) {
        String customerId = env.getArgument("customerId");
        String keyword = env.getArgument("keyword");
        List<Contact> contacts = contactRepository.selectList(null);
        if (customerId != null && !customerId.isBlank()) {
            contacts = contacts.stream().filter(c -> c.getCustomerId() != null
                    && c.getCustomerId().toString().equals(customerId)).toList();
        }
        if (keyword != null && !keyword.isBlank()) {
            String kw = keyword.toLowerCase();
            contacts = contacts.stream().filter(c ->
                    (c.getFirstName() != null && c.getFirstName().toLowerCase().contains(kw)) ||
                    (c.getLastName() != null && c.getLastName().toLowerCase().contains(kw)) ||
                    (c.getEmail() != null && c.getEmail().toLowerCase().contains(kw)) ||
                    (c.getPhone() != null && c.getPhone().contains(kw))
            ).toList();
        }
        List<Map<String, Object>> edges = contacts.stream().map(contact -> {
            Map<String, Object> edge = new LinkedHashMap<>();
            edge.put("node", contactToMap(contact));
            edge.put("cursor", contact.getId().toString());
            return edge;
        }).toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("edges", edges);
        result.put("totalCount", contacts.size());
        Map<String, Object> pageInfo = new LinkedHashMap<>();
        pageInfo.put("hasNextPage", false);
        pageInfo.put("hasPreviousPage", false);
        result.put("pageInfo", pageInfo);
        return result;
    }

    private Map<String, Object> leadToMap(Lead lead) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("__typename", "Lead");
        map.put("id", lead.getId().toString());
        map.put("leadNo", lead.getLeadNo());
        map.put("name", lead.getName());
        map.put("company", lead.getCompany());
        map.put("email", lead.getEmail());
        map.put("phone", lead.getPhone());
        map.put("source", lead.getSource());
        map.put("status", lead.getStatus());
        map.put("score", lead.getScore());
        map.put("industry", lead.getIndustry());
        map.put("city", lead.getCity());
        map.put("description", lead.getDescription());
        map.put("assignedTo", lead.getAssignedTo());
        map.put("createdAt", lead.getCreatedAt() != null ? lead.getCreatedAt().toString() : null);
        map.put("updatedAt", lead.getUpdatedAt() != null ? lead.getUpdatedAt().toString() : null);
        return map;
    }

    private Map<String, Object> opportunityToMap(Opportunity opp) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("__typename", "Opportunity");
        map.put("id", opp.getId().toString());
        map.put("opportunityNo", opp.getOpportunityNo());
        map.put("title", opp.getTitle());
        map.put("leadId", opp.getLeadId() != null ? opp.getLeadId().toString() : null);
        map.put("customerId", opp.getCustomerId() != null ? opp.getCustomerId().toString() : null);
        map.put("stage", opp.getStage());
        map.put("amount", opp.getAmount() != null ? opp.getAmount().doubleValue() : null);
        map.put("probability", opp.getProbability());
        map.put("expectedCloseDate", opp.getExpectedCloseDate() != null ? opp.getExpectedCloseDate().toString() : null);
        map.put("actualCloseDate", opp.getActualCloseDate() != null ? opp.getActualCloseDate().toString() : null);
        map.put("competitor", opp.getCompetitor());
        map.put("description", opp.getDescription());
        map.put("assignedTo", opp.getAssignedTo());
        map.put("createdAt", opp.getCreatedAt() != null ? opp.getCreatedAt().toString() : null);
        map.put("updatedAt", opp.getUpdatedAt() != null ? opp.getUpdatedAt().toString() : null);
        return map;
    }

    private Map<String, Object> ticketToMap(CustomerServiceTicket ticket) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("__typename", "Ticket");
        map.put("id", ticket.getId().toString());
        map.put("ticketNo", ticket.getTicketNo());
        map.put("title", ticket.getTitle());
        map.put("customerId", ticket.getCustomerId() != null ? ticket.getCustomerId().toString() : null);
        map.put("orderId", ticket.getOrderId() != null ? ticket.getOrderId().toString() : null);
        map.put("type", ticket.getType());
        map.put("priority", ticket.getPriority());
        map.put("status", ticket.getStatus());
        map.put("assignedTo", ticket.getAssignedTo());
        map.put("description", ticket.getDescription());
        map.put("resolution", ticket.getResolution());
        map.put("createdAt", ticket.getCreatedAt() != null ? ticket.getCreatedAt().toString() : null);
        map.put("updatedAt", ticket.getUpdatedAt() != null ? ticket.getUpdatedAt().toString() : null);
        return map;
    }

    private Map<String, Object> campaignToMap(Campaign campaign) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("__typename", "Campaign");
        map.put("id", campaign.getId().toString());
        map.put("name", campaign.getName());
        map.put("description", campaign.getDescription());
        map.put("type", campaign.getType());
        map.put("status", campaign.getStatus());
        map.put("startDate", campaign.getStartDate() != null ? campaign.getStartDate().toString() : null);
        map.put("endDate", campaign.getEndDate() != null ? campaign.getEndDate().toString() : null);
        map.put("budget", campaign.getBudget() != null ? campaign.getBudget().doubleValue() : null);
        map.put("actualCost", campaign.getActualCost() != null ? campaign.getActualCost().doubleValue() : null);
        map.put("expectedRevenue", campaign.getExpectedRevenue() != null ? campaign.getExpectedRevenue().doubleValue() : null);
        map.put("leadsGenerated", campaign.getLeadsGenerated());
        map.put("convertedCustomers", campaign.getConvertedCustomers());
        map.put("createdAt", campaign.getCreatedAt() != null ? campaign.getCreatedAt().toString() : null);
        map.put("updatedAt", campaign.getUpdatedAt() != null ? campaign.getUpdatedAt().toString() : null);
        return map;
    }

    private Map<String, Object> contactToMap(Contact contact) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("__typename", "Contact");
        map.put("id", contact.getId().toString());
        map.put("firstName", contact.getFirstName());
        map.put("lastName", contact.getLastName());
        map.put("email", contact.getEmail());
        map.put("phone", contact.getPhone());
        map.put("mobile", contact.getMobile());
        map.put("position", contact.getPosition());
        map.put("department", contact.getDepartment());
        map.put("customerId", contact.getCustomerId() != null ? contact.getCustomerId().toString() : null);
        map.put("isPrimary", contact.getIsPrimary());
        map.put("city", contact.getCity());
        map.put("createdAt", contact.getCreatedAt() != null ? contact.getCreatedAt().toString() : null);
        map.put("updatedAt", contact.getUpdatedAt() != null ? contact.getUpdatedAt().toString() : null);
        return map;
    }
}
