package com.metawebthree.crm.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
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
import lombok.Data;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import com.metawebthree.crm.interfaces.graphql.dto.Connection;
import com.metawebthree.crm.interfaces.graphql.dto.Edge;
import com.metawebthree.crm.interfaces.graphql.dto.PageInfo;

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
        if (representations == null || representations.isEmpty()) return null;
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        Object idObj = representation.get("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf(idObj.toString());
        return resolveRepresentation(typeName, id);
    }

    private Object resolveRepresentation(String typeName, Long id) {
        return switch (typeName) {
            case "Lead" -> nullableDto(leadQueryService.getById(id), LeadDTO::from);
            case "Opportunity" -> nullableDto(opportunityQueryService.getById(id), OpportunityDTO::from);
            case "Ticket" -> nullableDto(ticketQueryService.getById(id), TicketDTO::from);
            case "Campaign" -> nullableDto(campaignRepository.selectById(id), CampaignDTO::from);
            case "Contact" -> nullableDto(contactRepository.selectById(id), ContactDTO::from);
            default -> null;
        };
    }

    private <T, R> R nullableDto(T entity, java.util.function.Function<T, R> mapper) {
        return entity != null ? mapper.apply(entity) : null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof TypedNode node) {
            return env.getSchema().getObjectType(node.__typename());
        }
        if (src instanceof Map<?, ?> map) {
            return env.getSchema().getObjectType((String) map.get("__typename"));
        }
        return null;
    }

    private LeadDTO leadDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(leadQueryService.getById(argId(env)), LeadDTO::from);
    }

    private Connection<LeadDTO> leadsDataFetcher(DataFetchingEnvironment env) {
        List<Lead> leads = resolveLeads(env);
        return pageResult(leads, LeadDTO::from, l -> l.getId().toString());
    }

    private OpportunityDTO opportunityDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(opportunityQueryService.getById(argId(env)), OpportunityDTO::from);
    }

    private Connection<OpportunityDTO> opportunitiesDataFetcher(DataFetchingEnvironment env) {
        List<Opportunity> opportunities = resolveOpportunities(env);
        return pageResult(opportunities, OpportunityDTO::from, o -> o.getId().toString());
    }

    private TicketDTO ticketDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(ticketQueryService.getById(argId(env)), TicketDTO::from);
    }

    private Connection<TicketDTO> ticketsDataFetcher(DataFetchingEnvironment env) {
        List<CustomerServiceTicket> tickets = resolveTickets(env);
        return pageResult(tickets, TicketDTO::from, t -> t.getId().toString());
    }

    private CampaignDTO campaignDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(campaignRepository.selectById(argId(env)), CampaignDTO::from);
    }

    private Connection<CampaignDTO> campaignsDataFetcher(DataFetchingEnvironment env) {
        List<Campaign> campaigns = resolveCampaigns(env);
        return pageResult(campaigns, CampaignDTO::from, c -> c.getId().toString());
    }

    private ContactDTO contactDataFetcher(DataFetchingEnvironment env) {
        return nullableDto(contactRepository.selectById(argId(env)), ContactDTO::from);
    }

    private Connection<ContactDTO> contactsDataFetcher(DataFetchingEnvironment env) {
        List<Contact> contacts = resolveContacts(env);
        return pageResult(contacts, ContactDTO::from, c -> c.getId().toString());
    }

    private Long argId(DataFetchingEnvironment env) {
        Object id = env.getArgument("id");
        if (id == null) return null;
        if (id instanceof Number) return ((Number) id).longValue();
        return Long.valueOf(id.toString());
    }

    private List<Lead> resolveLeads(DataFetchingEnvironment env) {
        String keyword = env.getArgument("keyword");
        String status = env.getArgument("status");
        String source = env.getArgument("source");
        if (hasText(keyword)) return leadQueryService.search(keyword);
        if (hasText(status)) return leadQueryService.listByStatus(status);
        if (hasText(source)) return leadQueryService.listBySource(source);
        return leadQueryService.listAll();
    }

    private List<Opportunity> resolveOpportunities(DataFetchingEnvironment env) {
        String keyword = env.getArgument("keyword");
        String stage = env.getArgument("stage");
        if (hasText(keyword)) return opportunityQueryService.search(keyword);
        if (hasText(stage)) return opportunityQueryService.listByStage(stage);
        return opportunityQueryService.listAll();
    }

    private List<CustomerServiceTicket> resolveTickets(DataFetchingEnvironment env) {
        String priority = env.getArgument("priority");
        String status = env.getArgument("status");
        if (hasText(priority)) return ticketQueryService.listByPriority(priority);
        if (hasText(status)) return ticketQueryService.listByStatus(status);
        return ticketQueryService.listAll();
    }

    private List<Campaign> resolveCampaigns(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        String type = env.getArgument("type");
        LambdaQueryWrapper<Campaign> wrapper = new LambdaQueryWrapper<>();
        if (hasText(status)) wrapper.eq(Campaign::getStatus, status);
        if (hasText(type)) wrapper.eq(Campaign::getType, type);
        return campaignRepository.selectList(wrapper);
    }

    private List<Contact> resolveContacts(DataFetchingEnvironment env) {
        String customerId = env.getArgument("customerId");
        String keyword = env.getArgument("keyword");
        LambdaQueryWrapper<Contact> wrapper = new LambdaQueryWrapper<>();
        if (hasText(customerId)) wrapper.eq(Contact::getCustomerId, Long.valueOf(customerId));
        if (hasText(keyword)) {
            wrapper.and(w -> w.like(Contact::getFirstName, keyword)
                    .or().like(Contact::getLastName, keyword)
                    .or().like(Contact::getEmail, keyword)
                    .or().like(Contact::getPhone, keyword));
        }
        return contactRepository.selectList(wrapper);
    }

    private static boolean hasText(String s) {
        return s != null && !s.isBlank();
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }

    private <T, R> Connection<R> pageResult(List<T> items,
            java.util.function.Function<T, R> mapper,
            java.util.function.Function<T, String> cursorFn) {
        List<Edge<R>> edges = items.stream()
                .map(item -> new Edge<>(cursorFn.apply(item), mapper.apply(item)))
                .toList();
        return new Connection<>(edges, items.size(), new PageInfo(false, null));
    }

    private interface TypedNode {
        String __typename();
    }

    @Data
    public static class LeadDTO implements TypedNode {
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

        static LeadDTO from(Lead lead) {
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
    }

    @Data
    public static class OpportunityDTO implements TypedNode {
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

        static OpportunityDTO from(Opportunity opp) {
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
    }

    @Data
    public static class TicketDTO implements TypedNode {
        String __typename;
        String id;
        String ticketNo;
        String title;
        String customerId;
        String orderId;
        String type;
        String priority;
        String status;
        String assignedTo;
        String description;
        String resolution;
        String createdAt;
        String updatedAt;

        @Override
        public String __typename() { return __typename; }

        static TicketDTO from(CustomerServiceTicket ticket) {
            TicketDTO dto = new TicketDTO();
            dto.__typename = "Ticket";
            baseFields(dto, ticket);
            relationFields(dto, ticket);
            ticketFields(dto, ticket);
            timestampFields(dto, ticket);
            return dto;
        }

        private static void baseFields(TicketDTO dto, CustomerServiceTicket ticket) {
            dto.id = ticket.getId().toString();
            dto.ticketNo = ticket.getTicketNo();
            dto.title = ticket.getTitle();
        }

        private static void relationFields(TicketDTO dto, CustomerServiceTicket ticket) {
            dto.customerId = ticket.getCustomerId() != null ? ticket.getCustomerId().toString() : null;
            dto.orderId = ticket.getOrderId() != null ? ticket.getOrderId().toString() : null;
        }

        private static void ticketFields(TicketDTO dto, CustomerServiceTicket ticket) {
            dto.type = ticket.getType();
            dto.priority = ticket.getPriority();
            dto.status = ticket.getStatus();
            dto.assignedTo = ticket.getAssignedTo();
            dto.description = ticket.getDescription();
            dto.resolution = ticket.getResolution();
        }

        private static void timestampFields(TicketDTO dto, CustomerServiceTicket ticket) {
            dto.createdAt = str(ticket.getCreatedAt());
            dto.updatedAt = str(ticket.getUpdatedAt());
        }
    }

    @Data
    public static class CampaignDTO implements TypedNode {
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

        static CampaignDTO from(Campaign campaign) {
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
    }

    @Data
    public static class ContactDTO implements TypedNode {
        String __typename;
        String id;
        String firstName;
        String lastName;
        String email;
        String phone;
        String mobile;
        String position;
        String department;
        String customerId;
        Boolean isPrimary;
        String city;
        String createdAt;
        String updatedAt;

        @Override
        public String __typename() { return __typename; }

        static ContactDTO from(Contact contact) {
            ContactDTO dto = new ContactDTO();
            dto.__typename = "Contact";
            baseFields(dto, contact);
            orgFields(dto, contact);
            locationFields(dto, contact);
            timestampFields(dto, contact);
            return dto;
        }

        private static void baseFields(ContactDTO dto, Contact contact) {
            dto.id = contact.getId().toString();
            dto.firstName = contact.getFirstName();
            dto.lastName = contact.getLastName();
            dto.email = contact.getEmail();
            dto.phone = contact.getPhone();
            dto.mobile = contact.getMobile();
        }

        private static void orgFields(ContactDTO dto, Contact contact) {
            dto.position = contact.getPosition();
            dto.department = contact.getDepartment();
            dto.customerId = contact.getCustomerId() != null ? contact.getCustomerId().toString() : null;
            dto.isPrimary = contact.getIsPrimary();
        }

        private static void locationFields(ContactDTO dto, Contact contact) {
            dto.city = contact.getCity();
        }

        private static void timestampFields(ContactDTO dto, Contact contact) {
            dto.createdAt = str(contact.getCreatedAt());
            dto.updatedAt = str(contact.getUpdatedAt());
        }
    }
}
